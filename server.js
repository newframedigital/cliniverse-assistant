// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

const PORT = process.env.PORT || 8787;

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const upload = multer({ storage: multer.memoryStorage() });
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ---------------- Retry helper ---------------- */
async function withRetry(fn, { retries = 5, baseMs = 600 } = {}) {
  let lastErr;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      const status = e?.status || e?.response?.status;
      const retriable = [429, 500, 502, 503, 504].includes(status);
      if (!retriable || i === retries - 1) throw e;
      const delay = baseMs * Math.pow(2, i);
      console.warn(`Retry ${i + 1}/${retries} after ${status}. Waiting ${delay}ms...`);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

/* ---------------- Health ---------------- */
app.get('/ping', (_req, res) => res.json({ ok: true }));
app.get('/', (_req, res) => res.send('Cliniverse Assistant API OK'));

/* ---------------- 1) Create Vector Store ---------------- */
app.all('/init-vector-store', async (_req, res) => {
  try {
    const vs = await client.vectorStores.create({ name: 'cliniverse-kb' });
    res.json({ vector_store_id: vs.id });
  } catch (e) {
    console.error('INIT VS ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

/* ---------------- 2) Ingest: upload + attach ---------------- */
app.post('/ingest', upload.array('files'), async (req, res) => {
  try {
    const vId = (process.env.VECTOR_STORE_ID || '').trim();
    if (!vId) return res.status(400).json({ error: 'VECTOR_STORE_ID missing in .env' });
    if (!req.files?.length)
      return res.status(400).json({ error: 'No files attached. Field name must be "files".' });

    const norm = (s) => String(s || '').trim().replace(/\s+/g, '-');
    const prof = norm(req.body.profession).toLowerCase();
    const reg = norm(req.body.region).toUpperCase();
    const top = norm(req.body.topic).toLowerCase();
    const upd = norm(req.body.updated);
    const prefix = [prof, reg, top, upd].filter(Boolean).join('_');

    const results = [];
    for (const f of req.files) {
      const stampedName = prefix ? `${prefix}__${f.originalname}` : f.originalname;
      const tmpPath = path.join('/tmp', stampedName);
      fs.writeFileSync(tmpPath, f.buffer);

      const uploaded = await withRetry(() =>
        client.files.create({ file: fs.createReadStream(tmpPath), purpose: 'assistants' }),
      );
      console.log('UPLOADED:', stampedName, '->', uploaded.id);

      const attached = await withRetry(() =>
        client.vectorStores.files.create(vId, { file_id: uploaded.id }),
      );
      console.log('ATTACHED:', attached.id);

      try {
        fs.unlinkSync(tmpPath);
      } catch {}
      results.push({ stamped_name: stampedName, file_id: uploaded.id, vs_file_id: attached.id });
    }

    res.json({
      status: 'uploaded',
      count: results.length,
      items: results,
      applied_tags: { profession: prof, region: reg, topic: top, updated: upd },
    });
  } catch (e) {
    console.error('INGEST ERROR:', e);
    res.status(500).json({ error: e.message || 'Ingest failed' });
  }
});

/* ---------------- Helpers for SDK signature quirks ---------------- */
// Threads
async function threadCreate() {
  try {
    const t = await client.beta.threads.create(); // positional-friendly (no args)
    return t;
  } catch {
    const t = await client.beta.threads.create({}); // object fallback
    return t;
  }
}

// Messages.create
async function messageCreate(threadId, body) {
  try {
    const r = await client.beta.threads.messages.create(threadId, body);
    return r;
  } catch {
    const r = await client.beta.threads.messages.create({ thread_id: threadId, ...body });
    return r;
  }
}

// Messages.list
async function messageList(threadId, opts = {}) {
  try {
    const r = await client.beta.threads.messages.list(threadId, opts);
    return r;
  } catch {
    const r = await client.beta.threads.messages.list({ thread_id: threadId, ...opts });
    return r;
  }
}

// Runs.create
async function runCreate(threadId, body) {
  try {
    const r = await client.beta.threads.runs.create(threadId, body);
    return r;
  } catch {
    const r = await client.beta.threads.runs.create({ thread_id: threadId, ...body });
    return r;
  }
}

// Runs.retrieve
async function runRetrieve(threadId, runId) {
  try {
    const r = await client.beta.threads.runs.retrieve({ thread_id: threadId, run_id: runId });
    return r;
  } catch {
    const r = await client.beta.threads.runs.retrieve(threadId, runId);
    return r;
  }
}

// Runs.list
async function runList(threadId, opts = {}) {
  try {
    const r = await client.beta.threads.runs.list(threadId, opts);
    return r;
  } catch {
    const r = await client.beta.threads.runs.list({ thread_id: threadId, ...opts });
    return r;
  }
}

/* ---------------- Assistant bootstrap ---------------- */
const baseInstructions = `
You are Cliniverse Coach — a compliant, marketing-only assistant for physio, chiro, osteo, and RMT clinics in Canada and the US.

Voice & approach:
- Warm, concise, encouraging teacher. Never scold; motivate.
- Always structure copy reviews as:
  1) Verdict: "Compliant" or "Not compliant".
  2) Why: cite the exact rule in plain language (no legalese).
  3) Fixes: 2–3 compliant rewrites in the user’s tone.
  4) Tips: 2–4 quick, practical guardrails to remember next time.
- Do NOT include references/footnotes unless the user asks.
- Do NOT hand off to humans unless the user is clearly frustrated or explicitly asks for human help. Otherwise, you keep helping.

Guardrails (never output):
- Superlatives or superiority claims (“expert”, “best”, “leading”), guarantees/cures, unverifiable outcome claims, testimonials/endorsements (unless user states they’re permitted for their regulator), or anything that conflicts with local rules.
- Clinical advice. You only help with marketing/advertising compliance and copy.

Behavior when info is missing:
- If BOTH profession and province/state are unknown for the current session, ask once, in one friendly sentence, for both together — then stop and wait.
- If only one is missing, ask for just that one.
- If both are known, proceed without asking again and tailor to the known profession+region.
`.trim();

async function ensureAssistant() {
  const vId = (process.env.VECTOR_STORE_ID || '').trim();
  let aId = (process.env.ASSISTANT_ID || '').trim();

  if (aId) {
    try {
      const updated = await client.beta.assistants.update(aId, {
        model: 'gpt-4.1-mini',
        tools: [{ type: 'file_search' }],
        tool_resources: vId ? { file_search: { vector_store_ids: [vId] } } : undefined,
        instructions: baseInstructions,
      });
      return updated.id;
    } catch (e) {
      console.warn('Assistant update failed; creating new. Reason:', e?.message || e);
    }
  }

  try {
    const created = await client.beta.assistants.create({
      name: 'Cliniverse Coach',
      model: 'gpt-4.1-mini',
      tools: [{ type: 'file_search' }],
      tool_resources: vId ? { file_search: { vector_store_ids: [vId] } } : undefined,
      instructions: baseInstructions,
    });
    return created.id;
  } catch (e) {
    console.warn('assistant.create with tool_resources failed; retrying without. Reason:', e?.message || e);
    const created2 = await client.beta.assistants.create({
      name: 'Cliniverse Coach',
      model: 'gpt-4.1-mini',
      tools: [{ type: 'file_search' }],
      instructions: baseInstructions,
    });
    return created2.id;
  }
}

/* ---------------- Lightweight session memory ---------------- */
const SESSIONS = new Map(); // key: X-Session-Id (front-end), value: { profession, region }

const PROF_MAP = new Map([
  [/physio|physiotherap/i, 'physio'],
  [/chiro|chiropract/i, 'chiro'],
  [/osteo(?!por)/i, 'osteo'],
  [/\b(rmt|massage therapist|registered massage)\b/i, 'rmt'],
]);

const PROVINCE_ALIASES = {
  // Canada
  ON: [/ontario|^on\b/i],
  QC: [/quebec|québec|^qc\b/i],
  BC: [/\bbc\b|british columbia/i],
  AB: [/alberta|^ab\b/i],
  MB: [/manitoba|^mb\b/i],
  SK: [/saskatchewan|^sk\b/i],
  NS: [/nova scotia|^ns\b/i],
  NB: [/new brunswick|^nb\b/i],
  NL: [/newfoundland|labrador|^nl\b/i],
  PE: [/prince edward island|^pe\b/i],
  NT: [/northwest territories|^nt\b/i],
  YT: [/yukon|^yt\b/i],
  NU: [/nunavut|^nu\b/i],
  // US (sample)
  NY: [/new york|^ny\b/i],
  CA: [/california|^ca\b/i],
  TX: [/texas|^tx\b/i],
  FL: [/florida|^fl\b/i],
};

function extractProfession(text) {
  for (const [re, norm] of PROF_MAP) if (re.test(text)) return norm;
  return null;
}
function extractRegion(text) {
  for (const abbr of Object.keys(PROVINCE_ALIASES)) {
    if (PROVINCE_ALIASES[abbr].some((re) => re.test(text))) return abbr;
  }
  return null;
}

/* ---------------- 3) Chat ---------------- */
app.post('/chat', async (req, res) => {
  try {
    let { message, showDisclaimer = false } = req.body;

    // Session memory keyed by X-Session-Id header (front-end should send it)
    const sessionId = req.get('X-Session-Id') || req.ip || String(Date.now());
    if (!SESSIONS.has(sessionId)) SESSIONS.set(sessionId, { profession: null, region: null });
    const mem = SESSIONS.get(sessionId);

    // Update memory from this message if user volunteered info
    const profFromMsg = extractProfession(message);
    const regFromMsg = extractRegion(message);
    if (profFromMsg) mem.profession = profFromMsg;
    if (regFromMsg) mem.region = regFromMsg;

    // Known context
    const knownBits = [];
    if (mem.profession) knownBits.push(`profession=${mem.profession}`);
    if (mem.region) knownBits.push(`region=${mem.region}`);
    const contextLine = knownBits.length
      ? `Known user context: ${knownBits.join(', ')}`
      : `Known user context: none`;

    if (showDisclaimer) {
      message += `

Disclaimer: It is the practitioner’s responsibility to ensure marketing is accurate, verifiable, and compliant. Cliniverse provides guidance only.`;
    }

    // 1) Thread
    const thread = await threadCreate();
    const threadId = thread.id;

    // 2) User message
    await messageCreate(threadId, { role: 'user', content: message });

    // 3) Dynamic per-run instructions
    const dynamicInstructions = `
${contextLine}

If information is missing:
- If BOTH profession and province/state are missing, ask once in a single friendly sentence for both, and then wait for the reply.
- If only one is missing, ask for just that one, once, and then wait.
- If both are known, do NOT ask again; proceed.

When reviewing or generating copy, ALWAYS structure your answer:
1) Verdict: Compliant / Not compliant.
2) Why: explain briefly in plain language.
3) Fixes: provide 2–3 compliant rewrites in the user's tone.
4) Tips: 2–4 practical guardrails.

No references unless asked. No unsolicited hand-offs to humans. Only hand off if the user is clearly frustrated or explicitly asks for human help.
Avoid superlatives or superiority claims, guarantees/cures, unverifiable outcome claims, and testimonials unless the user confirms they’re permitted by their regulator.
`.trim();

    const vId = (process.env.VECTOR_STORE_ID || '').trim();
    const aId = await ensureAssistant();

    // 4) Create run (bind vector store if available)
    let run;
    try {
      run = await runCreate(threadId, {
        assistant_id: aId,
        ...(vId ? { tool_resources: { file_search: { vector_store_ids: [vId] } } } : {}),
        instructions: dynamicInstructions,
      });
    } catch {
      run = await runCreate(threadId, { assistant_id: aId, instructions: dynamicInstructions });
    }
    const runId = run.id;

    // 5) Poll
    const terminal = new Set(['completed', 'failed', 'cancelled', 'expired']);
    while (!terminal.has(run.status)) {
      await new Promise((r) => setTimeout(r, 900));
      try {
        run = await runRetrieve(threadId, runId);
      } catch {
        const list = await runList(threadId);
        run = list?.data?.find((r) => r.id === runId) || run;
      }
    }
    if (run.status !== 'completed') {
      return res.status(500).json({ error: `Run not completed: ${run.status}` });
    }

    // 6) Read reply
    const msgs = await messageList(threadId, { limit: 10 });
    const lastAssistantMsg = msgs.data.find((m) => m.role === 'assistant');
    const text = (lastAssistantMsg?.content || [])
      .map((c) => (c.type === 'text' ? c.text.value : ''))
      .join('\n')
      .trim();

    // Light clean (avoid literal placeholder leaking)
    const cleaned = text.replace(/\[Cliniverse Support Line\]/gi, 'our support team');

    res.json({ answer: cleaned || '(no response text)', thread_id: threadId, run_id: runId });
  } catch (e) {
    console.error('CHAT ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Cliniverse Assistant running on http://localhost:${PORT}`);
});

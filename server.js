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
You are Cliniverse Coach — a friendly, expert marketing and compliance assistant for physiotherapy, chiropractic, osteopathy, and RMT clinics across Canada and the US.

Your voice:
- Professional yet friendly, conversational, supportive, and human.
- Write like a seasoned marketing mentor — clear, motivating, and approachable.
- Always encourage the user and make them feel confident and capable.

Your purpose:
- Help clinicians create compliant marketing and advertising materials.
- Educate naturally — teach through explanation, not rigid structure.
- Make compliance sound empowering, not restrictive.

Tone guidelines:
- Warm and encouraging (“Here’s how we can make this even better…”).
- No robotic phrases like “Verdict” or numbered sections.
- Be short and to the point, but personal and motivating.

Core compliance rules (never break these):
- Never use guarantees, superiority claims, or unverifiable results.
- No testimonials or comparisons to other providers.
- Focus on accuracy, honesty, and patient education.
- Always align with professional advertising guidelines for their province/state by internally referencing the verified policy PDFs in the vector library and checking publicly available regulator sources — but never mention these references to the user.
- If unsure or unclear about a rule, always default to the stricter and more conservative interpretation to protect compliance.
- Do not use titles or superlatives like “expert,” “best,” “leading,” “top-rated,” “world-class,” “guaranteed,” or “cure.” Prefer neutral, accurate phrasing such as “regulated chiropractor,” “experienced in,” “focused on,” or “additional training in [area].”
- Once profession and province/state are known, do not restate them in every reply unless the user changes context.

If the user’s profession or region is unknown, ask for both once in a friendly, single sentence:
“Before I tailor this, could you tell me your profession (physio, chiro, osteo, or RMT) and which province or state you’re in?”

Always align with the local professional advertising guidelines for the user’s province/state, and if uncertain, choose the safer, more conservative wording that clearly avoids superlatives and incentives.

If the user seems frustrated, respond with calm reassurance, kindness, and encouragement. Only mention contacting Tash & Tyler if the user is truly upset or asks for human help.
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
// --- Minimal compliance guardrail (tone-safe) ---
function sanitizeMarketing(text, { profession, region } = {}) {
  if (!text) return text;

  // 1) Superlatives we never want in regulated health marketing
  const superlatives = /\b(best|#1|number\s*1|leading|top[-\s]?rated|expert|world[-\s]?class|state[-\s]?of[-\s]?the[-\s]?art)\b/gi;
  text = text.replace(superlatives, (m) => {
    // friendlier, compliant swaps
    if (/expert/i.test(m)) return 'experienced';
    if (/leading|top|best|#1|number\s*1/i.test(m)) return 'trusted';
    if (/world|state[-\s]?of[-\s]?the[-\s]?art/i.test(m)) return 'professional';
    return 'professional';
  });

  // 2) Incentives / inducements (free, % off, coupons) – block outright
  const incentives = /\b(free|complimentary|no[-\s]?cost|% ?off|percent ?off|discount|coupon|deal|special\s+offer|two[-\s]?for[-\s]?one|bogo)\b/gi;
  if (incentives.test(text)) {
    text = text.replace(incentives, 'introductory'); // soften wording
    // also scrub explicit numerics like "50% off"
    text = text.replace(/\b\d{1,3}\s?% ?off\b/gi, 'introductory rate');
  }

  // 3) Gentle advisory if we changed anything
  if (superlatives.test || incentives.test) {
    // Keep one calm, friendly note at the end (single sentence)
    if (!/Compliance note:/i.test(text)) {
      text += '\n\nCompliance note: I kept the wording conservative to avoid superlatives or promotional incentives.';
    }
  }
  return text;
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

Respond like a marketing mentor who genuinely wants to help clinicians succeed.
Speak naturally — like a conversation — while making sure every suggestion is compliant for their profession and province/state.

If both profession and province/state are missing, ask for them once in a friendly way and then wait for the reply.
If one is missing, ask only for that one.
If both are known, just proceed normally.

Be conversational, positive, and teaching-focused.
When you give examples or edits, explain *why* briefly but naturally (e.g., “This keeps it compliant because it avoids promising outcomes”).
Offer helpful, simple alternatives and tips the user can apply right away.
Avoid rigid formatting or headings like “Verdict” or “Fixes.”
Avoid decorative asterisks or bold for emphasis. Use plain sentences. If a short list helps, use simple hyphen bullets and keep it to a maximum of 3 items.
Mention the user’s profession/region only when it directly affects the wording; otherwise omit it to avoid repetition.

Don’t repeat the user’s profession/region in every message; mention it sparingly.
Prefer short paragraphs; use bullet points only if the user asks for lists.
If you detect risky wording (superlatives like “expert/best/leading” or incentives like “free”, “% off”), rewrite it to compliant language and briefly say why (one friendly sentence).

If you’re uncertain about compliance, always take the safer, more conservative route in your recommendation — and phrase it as supportive guidance (e.g., “To stay on the safe side, here’s how I’d suggest framing it.”)

Never give medical or treatment advice. Stay purely on marketing, content, and communication.
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
// after you compute `text`
const { profession, region } = (req.body || {});
const safeAnswer = sanitizeMarketing(text, { profession, region });

res.json({
  answer: safeAnswer || '(no response text)',
  thread_id: threadId,
  run_id: runId
});
    res.json({ answer: cleaned || '(no response text)', thread_id: threadId, run_id: runId });
  } catch (e) {
    console.error('CHAT ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Cliniverse Assistant running on http://localhost:${PORT}`);
});

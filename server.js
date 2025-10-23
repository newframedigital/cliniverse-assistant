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
    try { return await fn(); }
    catch (e) {
      lastErr = e;
      const status = e?.status || e?.response?.status;
      const retriable = [429, 500, 502, 503, 504].includes(status);
      if (!retriable || i === retries - 1) throw e;
      const delay = baseMs * Math.pow(2, i);
      console.warn(`Retry ${i + 1}/${retries} after ${status}. Waiting ${delay}ms...`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

/* ---------------- Health ---------------- */
app.get('/ping', (_req, res) => res.json({ ok: true }));
app.get('/', (_req, res) => {
  res.send('Cliniverse Assistant API OK');
});

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
    if (!req.files?.length) return res.status(400).json({ error: 'No files attached. Field name must be "files".' });

    const norm = s => String(s || '').trim().replace(/\s+/g, '-');
    const prof = norm(req.body.profession).toLowerCase();
    const reg  = norm(req.body.region).toUpperCase();
    const top  = norm(req.body.topic).toLowerCase();
    const upd  = norm(req.body.updated);
    const prefix = [prof, reg, top, upd].filter(Boolean).join('_');

    const results = [];
    for (const f of req.files) {
      const stampedName = prefix ? `${prefix}__${f.originalname}` : f.originalname;
      const tmpPath = path.join('/tmp', stampedName);
      fs.writeFileSync(tmpPath, f.buffer);

      const uploaded = await withRetry(() =>
        client.files.create({ file: fs.createReadStream(tmpPath), purpose: 'assistants' })
      );
      console.log('UPLOADED:', stampedName, '->', uploaded.id);

      const attached = await withRetry(() =>
        client.vectorStores.files.create((process.env.VECTOR_STORE_ID || '').trim(), { file_id: uploaded.id })
      );
      console.log('ATTACHED:', attached.id);

      try { fs.unlinkSync(tmpPath); } catch {}
      results.push({ stamped_name: stampedName, file_id: uploaded.id, vs_file_id: attached.id });
    }

    res.json({ status: 'uploaded', count: results.length, items: results, applied_tags: { profession: prof, region: reg, topic: top, updated: upd } });
  } catch (e) {
    console.error('INGEST ERROR:', e);
    res.status(500).json({ error: e.message || 'Ingest failed' });
  }
});

/* ---------------- Helpers that adapt to SDK quirks ---------------- */
// Threads
async function threadCreate() {
  try {
    const t = await client.beta.threads.create(); // positional-friendly (no args)
    console.log('DEBUG threadCreate: positional ok');
    return t;
  } catch {
    const t = await client.beta.threads.create({});
    console.log('DEBUG threadCreate: object ok');
    return t;
  }
}

async function messageCreate(threadId, body) {
  try {
    const r = await client.beta.threads.messages.create(threadId, body);
    console.log('DEBUG messages.create: positional ok');
    return r;
  } catch {
    const r = await client.beta.threads.messages.create({ thread_id: threadId, ...body });
    console.log('DEBUG messages.create: object ok (fallback)');
    return r;
  }
}

async function messageList(threadId, opts = {}) {
  try {
    const r = await client.beta.threads.messages.list(threadId, opts);
    console.log('DEBUG messages.list: positional ok');
    return r;
  } catch {
    const r = await client.beta.threads.messages.list({ thread_id: threadId, ...opts });
    console.log('DEBUG messages.list: object ok (fallback)');
    return r;
  }
}

async function runCreate(threadId, body) {
  try {
    const r = await client.beta.threads.runs.create(threadId, body);
    console.log('DEBUG runs.create: positional ok');
    return r;
  } catch {
    const r = await client.beta.threads.runs.create({ thread_id: threadId, ...body });
    console.log('DEBUG runs.create: object ok (fallback)');
    return r;
  }
}

async function runRetrieve(threadId, runId) {
  try {
    const r = await client.beta.threads.runs.retrieve({ thread_id: threadId, run_id: runId });
    console.log('DEBUG runs.retrieve: object ok');
    return r;
  } catch {
    const r = await client.beta.threads.runs.retrieve(threadId, runId);
    console.log('DEBUG runs.retrieve: positional ok (fallback)');
    return r;
  }
}

async function runList(threadId, opts = {}) {
  try {
    const r = await client.beta.threads.runs.list(threadId, opts);
    console.log('DEBUG runs.list: positional ok');
    return r;
  } catch {
    const r = await client.beta.threads.runs.list({ thread_id: threadId, ...opts });
    console.log('DEBUG runs.list: object ok (fallback)');
    return r;
  }
}

/* ---------------- Assistant bootstrap ---------------- */
async function ensureAssistant() {
  const vId = (process.env.VECTOR_STORE_ID || '').trim();
  let aId = (process.env.ASSISTANT_ID || '').trim();

  const baseInstructions = `
You are Cliniverse Coach — a friendly, concise, marketing-only assistant for physio, chiro, osteo, and RMT clinics.
Think like a compliant Alex Hormozi: direct, practical, growth-minded, and within regulations.

Context & Memory
- Treat the conversation as a single session. Use the thread’s earlier messages as memory.
- If BOTH profession and province/state are missing, ask for them ONCE up front.
- If you can infer them from the thread, do not ask again.
- If something is still missing, still provide a helpful draft answer immediately with assumptions noted and then ask one focused follow-up question.

Compliance (hard rules; always apply to copy you produce)
- DO NOT use superiority or qualitative claims: “expert”, “best”, “top-rated”, “#1”, “leading”, “most advanced”, “state-of-the-art”, “superior”, etc.
- DO NOT guarantee outcomes or cures: “guarantee”, “cure”, “fix”, “permanent results”, etc.
- Be truthful, verifiable, and not misleading. No testimonials unless the jurisdiction explicitly allows and user confirms compliance steps.
- Avoid direct comparisons unless factual and verifiable.
- Accurately represent qualifications; no exaggeration.
- Respect privacy: no patient-identifiable details.
- Apply regional and profession-specific ad rules.

Self-check BEFORE replying
- Quickly scan your own draft for ANY violations of the above. If found, rewrite to remove or rephrase.
- If a user asks for a risky claim, provide a compliant alternative and briefly explain why.

Tone & UX
- Warm, encouraging, solutions-first. Avoid repeating the same question.
- If the user is frustrated, acknowledge it and move forward with best-effort guidance.
- Never output bracket placeholders or tool tags. Use plain text only.
- Do NOT append "References:" or cite source docs unless the user explicitly asks.

Output style
- Lead with the answer (e.g., the caption/headlines) tailored to what you know.
- Add 2–3 quick compliance tips inline when relevant.
- End with one short, optional follow-up question to refine further.

Optional support line
- If appropriate, you may add one short sentence like:
  "If you want extra help, Tash & Tyler at Cliniverse can support you with compliant marketing."
(Plain text only — no bracketed placeholders or links.)
`.trim();

  if (aId) {
    try {
      const updated = await client.beta.assistants.update(aId, {
        model: 'gpt-4.1-mini',
        tools: [{ type: 'file_search' }],
        tool_resources: (process.env.VECTOR_STORE_ID ? { file_search: { vector_store_ids: [vId] } } : undefined),
        instructions: baseInstructions
      });
      console.log('DEBUG assistant.update ok');
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
      tool_resources: (process.env.VECTOR_STORE_ID ? { file_search: { vector_store_ids: [vId] } } : undefined),
      instructions: baseInstructions
    });
    console.log('DEBUG assistant.create ok (with tool_resources if set)');
    return created.id;
  } catch (e) {
    console.warn('assistant.create with tool_resources failed; retrying without. Reason:', e?.message || e);
    const created2 = await client.beta.assistants.create({
      name: 'Cliniverse Coach',
      model: 'gpt-4.1-mini',
      tools: [{ type: 'file_search' }],
      instructions: baseInstructions
    });
    console.log('DEBUG assistant.create ok (without tool_resources)');
    return created2.id;
  }
}

/* ---------------- 2b) init-assistant ---------------- */
app.all('/init-assistant', async (_req, res) => {
  try {
    const aId = await ensureAssistant();
    res.json({ assistant_id: aId });
  } catch (e) {
    console.error('INIT ASSISTANT ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

/* ---------------- 3) Chat (thread-aware + compliance post-filter) ---------------- */
app.post('/chat', async (req, res) => {
  try {
    let { message: rawMessage, threadId, showDisclaimer = false } = req.body;
    let message = String(rawMessage || '').trim();

    const aId = await ensureAssistant();
    const vId = (process.env.VECTOR_STORE_ID || '').trim();

    if (showDisclaimer) {
      message += `

Disclaimer: It is the practitioner’s responsibility to ensure marketing is accurate, verifiable, and compliant. Cliniverse provides guidance only.`;
    }

    // 1) Thread (reuse if provided, else create)
    let threadIdFinal = (threadId || '').trim();
    if (!threadIdFinal) {
      const thread = await threadCreate();
      threadIdFinal = thread.id;
    }
    console.log('THREAD:', threadIdFinal);

    // 2) Message
    await messageCreate(threadIdFinal, { role: 'user', content: message });

    // 3) Run — bind vector store if present; fallback without
    let run;
    try {
      run = await runCreate(threadIdFinal, {
        assistant_id: aId,
        tool_resources: vId ? { file_search: { vector_store_ids: [vId] } } : undefined
      });
      console.log('DEBUG runs.create ok (with tool_resources if set)');
    } catch (e) {
      console.warn('runs.create with tool_resources failed; retrying without. Reason:', e?.message || e);
      run = await runCreate(threadIdFinal, { assistant_id: aId });
    }
    const runId = run.id;
    console.log('RUN:', runId);

    // 4) Poll
    const terminal = new Set(['completed', 'failed', 'cancelled', 'expired']);
    while (!terminal.has(run.status)) {
      await new Promise(r => setTimeout(r, 900));
      try {
        run = await runRetrieve(threadIdFinal, runId);
      } catch (err) {
        console.warn('runs.retrieve failed; trying runs.list fallback. Reason:', err?.message || err);
        const list = await runList(threadIdFinal);
        run = list?.data?.find(r => r.id === runId) || run;
      }
      console.log('STATUS:', run?.status || '(unknown)');
    }

    if (run.status !== 'completed') {
      return res.status(500).json({ error: `Run not completed: ${run.status}` });
    }

    // 5) Read reply
    const msgs = await messageList(threadIdFinal, { limit: 10 });
    const lastAssistantMsg = msgs.data.find(m => m.role === 'assistant');
    const text = (lastAssistantMsg?.content || [])
      .map(c => (c.type === 'text' ? c.text.value : ''))
      .join('\n')
      .trim();

    // ---- Compliance post-filter ----
    const bannedPhrases = /\b(expert|top[-\s]?rated|best\b|#1\b|leading\b|most advanced|state[-\s]?of[-\s]?the[-\s]?art|superior|guarantee|guaranteed|cure|permanent results|fix\b|fastest\b)\b/i;

    let safeText = text || '';
    if (bannedPhrases.test(safeText)) {
      try {
        const fix = await client.chat.completions.create({
          model: 'gpt-4.1-mini',
          temperature: 0.4,
          messages: [
            {
              role: 'system',
              content:
                'You are a compliance editor. Rewrite the user-provided marketing copy so it is compliant for healthcare advertising. ' +
                'Remove superiority/quality claims (expert/best/#1/leading/most advanced/state-of-the-art/superior), guarantees/cures, and anything misleading. ' +
                'Keep the intent and value, but use neutral, verifiable phrasing. Return only the revised copy.'
            },
            { role: 'user', content: safeText }
          ]
        });
        const revised = fix.choices?.[0]?.message?.content?.trim();
        if (revised) safeText = revised;
      } catch (err) {
        console.warn('Compliance fix pass failed; returning original text.', err?.message || err);
      }
    }

    res.json({ answer: safeText || '(no response text)', thread_id: threadIdFinal, run_id: runId });
  } catch (e) {
    console.error('CHAT ERROR:', e);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Cliniverse Assistant running on http://localhost:${PORT}`);
});

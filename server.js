// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

const PORT = process.env.PORT || 8787;
const ASSISTANT_ID = (process.env.ASSISTANT_ID || '').trim();
const VECTOR_STORE_ID = (process.env.VECTOR_STORE_ID || '').trim();

// Hard fail if IDs are not provided
if (!ASSISTANT_ID || !VECTOR_STORE_ID) {
  console.error('FATAL: ASSISTANT_ID and VECTOR_STORE_ID must be set in .env');
  process.exit(1);
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const upload = multer({ storage: multer.memoryStorage() });
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ---------------- Health ---------------- */
app.get('/ping', (_req, res) => res.json({ ok: true }));
app.get('/', (_req, res) => res.send('Cliniverse Assistant API OK'));

/* ---------------- Ingest: upload + attach to your single vector store ---------------- */
app.post('/ingest', upload.array('files'), async (req, res) => {
  try {
    if (!req.files?.length) {
      return res.status(400).json({ error: 'No files attached. Field name must be "files".' });
    }

    const norm = (s) => String(s || '').trim().replace(/\s+/g, '-');

    const prof = norm(req.body.profession).toLowerCase();
    const region = norm(req.body.region).toUpperCase();
    const topic = norm(req.body.topic).toLowerCase(); // e.g., ads, socialmedia, general
    const updated = norm(req.body.updated);
    const prefix = [prof, region, topic, updated].filter(Boolean).join('_');

    const results = [];
    for (const f of req.files) {
      const stampedName = prefix ? `${prefix}__${f.originalname}` : f.originalname;
      const tmpPath = path.join('/tmp', stampedName);
      fs.writeFileSync(tmpPath, f.buffer);

      const uploaded = await client.files.create({
        file: fs.createReadStream(tmpPath),
        purpose: 'assistants',
      });

      const attached = await client.vectorStores.files.create(VECTOR_STORE_ID, {
        file_id: uploaded.id,
      });

      try { fs.unlinkSync(tmpPath); } catch {}

      results.push({
        stamped_name: stampedName,
        file_id: uploaded.id,
        vs_file_id: attached.id,
      });
    }

    res.json({
      status: 'uploaded',
      count: results.length,
      items: results,
      applied_tags: { profession: prof, region, topic, updated },
    });
  } catch (e) {
    console.error('INGEST ERROR:', e);
    res.status(500).json({ error: e.message || 'Ingest failed' });
  }
});

/* ---------------- Helpers (no creates; we only use your fixed IDs) ---------------- */
// Create a new thread or reuse a provided one
async function ensureThread(threadId) {
  if (threadId) return { id: threadId };
  return client.beta.threads.create();
}

async function messageCreate(threadId, body) {
  // Try positional, then object style for SDK compatibility
  try {
    return await client.beta.threads.messages.create(threadId, body);
  } catch {
    return await client.beta.threads.messages.create({ thread_id: threadId, ...body });
  }
}

async function runCreate(threadId, body) {
  try {
    return await client.beta.threads.runs.create(threadId, body);
  } catch {
    return await client.beta.threads.runs.create({ thread_id: threadId, ...body });
  }
}

async function runRetrieve(threadId, runId) {
  try {
    return await client.beta.threads.runs.retrieve({ thread_id: threadId, run_id: runId });
  } catch {
    return await client.beta.threads.runs.retrieve(threadId, runId);
  }
}

async function runList(threadId, opts = {}) {
  try {
    return await client.beta.threads.runs.list(threadId, opts);
  } catch {
    return await client.beta.threads.runs.list({ thread_id: threadId, ...opts });
  }
}

async function messageList(threadId, opts = {}) {
  try {
    return await client.beta.threads.messages.list(threadId, opts);
  } catch {
    return await client.beta.threads.messages.list({ thread_id: threadId, ...opts });
  }
}

/* ---------------- Instruction blocks ---------------- */
/**
 * Base tone/guardrails — permanent traits (friendly, mentor, no jargon, no citations).
 * NOTE: We do NOT update the Assistant object — we pass instructions at run-time only.
 */
const baseInstructions = `
You are Cliniverse Coach — a friendly, expert marketing and compliance assistant for physiotherapy (including “physical therapy” / “PT”), chiropractic, osteopathy, and RMT clinics across Canada and the US.

Voice & style:
- Warm, encouraging, and conversational. Write like a supportive marketing mentor.
- Be concise and practical. Teach gently with short explanations and good examples.
- Avoid rigid headings or heavy lists. Prefer short paragraphs. Use simple hyphen bullets only when truly helpful.

Scope:
- Marketing/communications only. Never give clinical or treatment advice.

Core compliance stance (never break these):
- No guarantees, no superiority claims, no unverifiable results.
- No testimonials/endorsements or provider comparisons.
- No user-visible citations, links, or policy lists in answers unless explicitly asked.
- If you detect risky wording (e.g., “expert,” “best,” “leading,” “guaranteed,” “cure,” “free,” “% off,” “discount,” “complimentary”), rewrite to safe, neutral language.

Profession vocabulary:
- Treat “physical therapist”, “PT”, “physical therapy” as physiotherapy.
- Treat “registered massage therapist”, “massage therapist” as RMT.

Memory within a chat:
- If profession and province/state are unknown, ask for both once, in a single friendly sentence, and wait.
- If one is missing, ask only for that one, once, and wait.
- If both are known from earlier in the same thread, don’t ask again or repeat them unless the user changes them.

Teacher mindset:
- Answer the user’s request directly first (e.g., give the caption).
- Then add 1–2 brief teaching tips on why it’s compliant or how to improve it further.
- Keep the tone kind, motivating, and confidence-building.
`.trim();

/**
 * Retrieval policy — how to use the vector store and fallbacks.
 * We rely on file_search tool and filename cues to pick the best docs.
 * We never surface filenames or retrieval details to the user.
 */
function buildDynamicInstructions(userContextLine) {
  return `
${userContextLine}

Retrieval logic (vector-first → national → strict-safe):
1) Use file_search on the verified Vector Store. Prefer files for this exact profession + province/state (hint: filenames use patterns like \`{profession}_{country}_{province/state}_*\` where profession ∈ physio|chiro|osteo|rmt and country ∈ CA|US).
   - Search ALL documents for that province/state (ads/marketing/social/general/statutes/codes). Don’t stop at the first match; consider the whole set for a coherent answer.
2) If no clear provincial/state rule is found in the store, use national/federal guidance from the store for that profession and country (e.g., \`physio_CA_general_*\`, \`physio_US_general_*\`).
3) If neither provincial/state nor national documents exist in the store, answer conservatively with strict-safe defaults (no guarantees, no superlatives, no incentives/discounts/freebies, no testimonials/comparisons). Do NOT mention that sources are missing; simply write conservatively.

Communication rules:
- Do NOT include citations, filenames, links, or “references” unless the user asks.
- Avoid repeating the user’s profession/region in every message; only mention if it changes the wording.
- Avoid decorative asterisks or bold for emphasis; prefer clean sentences.
- If the user asks for copy (e.g., a caption/ad), produce it first, then give 1–2 short coaching tips.

If uncertain, choose the safer wording, and keep it friendly and helpful.
`.trim();
}

/* ---------------- Chat ---------------- */
app.post('/chat', async (req, res) => {
  try {
    let { message, threadId = null, showDisclaimer = false } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'Missing "message" (string)' });
    }

    // Optional disclaimer appended only on the first turn (let the frontend control this)
    if (showDisclaimer) {
      message += `

Disclaimer: It is the practitioner’s responsibility to ensure marketing is accurate, verifiable, and compliant. Cliniverse provides guidance only.`;
    }

    // 1) Ensure thread (new or reuse)
    const thread = await ensureThread(threadId);
    const realThreadId = thread.id;

    // 2) Add user message
    await messageCreate(realThreadId, { role: 'user', content: message });

    // 3) Build dynamic run-time instructions
    //    We let the model infer profession/region from the ongoing thread (memory),
    //    and guide it to use the vector store in the right order.
    const knownContext = `Known chat context is maintained within this OpenAI thread. Use prior user messages to remember profession and province/state for this session.`;
    const dynamicInstructions = [
      baseInstructions,
      buildDynamicInstructions(knownContext),
    ].join('\n\n');

    // 4) Create run (bind your vector store at run level)
    let run = await runCreate(realThreadId, {
      assistant_id: ASSISTANT_ID,
      instructions: dynamicInstructions,
      tool_resources: { file_search: { vector_store_ids: [VECTOR_STORE_ID] } },
    });

    // 5) Poll until completion
    const terminal = new Set(['completed', 'failed', 'cancelled', 'expired']);
    while (!terminal.has(run.status)) {
      await new Promise((r) => setTimeout(r, 900));
      try {
        run = await runRetrieve(realThreadId, run.id);
      } catch {
        const list = await runList(realThreadId, { limit: 5 });
        run = list?.data?.find((r) => r.id === run.id) || run;
      }
    }

    if (run.status !== 'completed') {
      return res.status(500).json({ error: `Run not completed: ${run.status}` });
    }

    // 6) Read the assistant reply
    const msgs = await messageList(realThreadId, { limit: 10 });
    const lastAssistantMsg = msgs.data.find((m) => m.role === 'assistant');
    const answer = (lastAssistantMsg?.content || [])
      .map((c) => (c.type === 'text' ? c.text.value : ''))
      .join('\n')
      .trim();

    return res.json({
      answer: answer || '(no response text)',
      thread_id: realThreadId,
      run_id: run.id,
    });
  } catch (e) {
    console.error('CHAT ERROR:', e);
    res.status(500).json({ error: e.message || 'Chat failed' });
  }
});

/* ---------------- Start ---------------- */
app.listen(PORT, () => {
  console.log(`Cliniverse Assistant running on http://localhost:${PORT}`);
  console.log(`Using Assistant: ${ASSISTANT_ID}`);
  console.log(`Using Vector Store: ${VECTOR_STORE_ID}`);
});

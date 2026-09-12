/**
 * CLARIFY backend — single-file version.
 *
 * Implements the 5 endpoints the frontend (clarify-mvp.html) already expects:
 *   POST /api/sessions
 *   POST /api/sessions/:id/upload
 *   POST /api/sessions/:id/text
 *   POST /api/sessions/:id/analyze
 *   POST /api/sessions/:id/ask
 *   POST /api/sessions/:id/end
 *
 * Sessions are held in memory only (a Map) — nothing is written to disk,
 * matching the "no storage" promise in the frontend's footer. Restarting
 * the server clears all sessions, which is fine for this use case.
 *
 * Requires an ANTHROPIC_API_KEY environment variable. Never put the key
 * in the frontend — this server is the only thing that ever sees it.
 */

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const crypto = require("crypto");
const Anthropic = require("@anthropic-ai/sdk");

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const PORT = process.env.PORT || 3001;
const MODEL = process.env.CLARIFY_MODEL || "claude-sonnet-4-6";

// Lock CORS down to your deployed frontend's origin in production.
const allowedOrigins = (process.env.ALLOWED_ORIGIN || "*").split(",").map(s => s.trim());
app.use(cors({
  origin: allowedOrigins.includes("*") ? true : allowedOrigins,
}));
app.use(express.json({ limit: "2mb" }));

// sessionId -> { text: string, analysis: object|null, createdAt: number }
const sessions = new Map();

// Sessions older than this are swept out so memory doesn't grow unbounded.
const SESSION_TTL_MS = 60 * 60 * 1000; // 1 hour
setInterval(() => {
  const now = Date.now();
  for (const [id, s] of sessions) {
    if (now - s.createdAt > SESSION_TTL_MS) sessions.delete(id);
  }
}, 10 * 60 * 1000).unref();

function requireSession(req, res, next) {
  const s = sessions.get(req.params.id);
  if (!s) return res.status(404).json({ error: "Session not found or expired." });
  req.session = s;
  next();
}

// ---- Session lifecycle ----

app.post("/api/sessions", (req, res) => {
  const id = crypto.randomUUID();
  sessions.set(id, { text: "", analysis: null, createdAt: Date.now() });
  res.json({ sessionId: id });
});

app.post("/api/sessions/:id/end", requireSession, (req, res) => {
  sessions.delete(req.params.id);
  res.json({ ok: true });
});

// ---- Getting the document text into the session ----

app.post("/api/sessions/:id/text", requireSession, (req, res) => {
  const { text } = req.body || {};
  if (!text || !text.trim()) return res.status(400).json({ error: "No text provided." });
  req.session.text = text;
  res.json({ ok: true, length: text.length });
});

app.post("/api/sessions/:id/upload", requireSession, upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded." });
  try {
    const text = await extractText(req.file);
    if (!text.trim()) return res.status(422).json({ error: "Couldn't read any text from that file." });
    req.session.text = text;
    res.json({ ok: true, length: text.length });
  } catch (err) {
    console.error("upload/extract error:", err);
    res.status(500).json({ error: "Failed to read that file." });
  }
});

async function extractText(file) {
  const name = (file.originalname || "").toLowerCase();
  if (name.endsWith(".txt")) {
    return file.buffer.toString("utf-8");
  }
  if (name.endsWith(".pdf")) {
    const pdfParse = require("pdf-parse");
    const data = await pdfParse(file.buffer);
    return data.text;
  }
  if (name.endsWith(".docx")) {
    const mammoth = require("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: file.buffer });
    return value;
  }
  throw new Error("Unsupported file type");
}

// ---- Analysis ----

const ANALYSIS_SYSTEM_PROMPT = `You are CLARIFY, an assistant that explains contracts in plain language for the person about to sign them, not the party who wrote them.

Read the contract text the user provides and respond with ONLY a single JSON object (no markdown fences, no commentary before or after) matching exactly this shape:

{
  "snapshot": { "summary": string, "parties": string, "term": string, "estCost": string },
  "gaps": string[],
  "findings": [
    { "title": string, "status": "Clearly stated" | "Inferred" | "Unclear" | "Worth checking",
      "explanation": string, "whyItMatters": string, "page": string, "clause": string,
      "quote": string, "question": string }
  ],
  "costs": [ { "label": string, "type": string, "amount": string } ],
  "dates": [ { "label": string, "date": string } ],
  "obligations": { "user": string[], "other": string[] },
  "checklist": string[],
  "suggestedQuestions": string[]
}

Rules:
- Every "quote" must be copied verbatim from the supplied text (a few words to one sentence), and "page"/"clause" should reference where in the document it appears as best you can tell (use "Not specified" if the document has no page/clause markers).
- Use "status" honestly: "Clearly stated" only when the contract says it outright; "Inferred" when you're reading between the lines; "Unclear" when the language is ambiguous; "Worth checking" for anything risky enough the user should ask about it before signing.
- Never invent numbers, dates, or obligations that aren't in the text. If something important is missing (e.g. no termination clause), list it in "gaps" instead of guessing.
- Write for someone with no legal background. Short, plain sentences.`;

app.post("/api/sessions/:id/analyze", requireSession, async (req, res) => {
  const text = req.session.text;
  if (!text || !text.trim()) return res.status(400).json({ error: "No document text to analyze yet." });

  try {
    const msg = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 4000,
      system: ANALYSIS_SYSTEM_PROMPT,
      messages: [{ role: "user", content: `Contract text:\n\n${text}` }],
    });
    const raw = msg.content.map(b => (b.type === "text" ? b.text : "")).join("");
    const analysis = parseJsonLoose(raw);
    req.session.analysis = analysis;
    res.json({ analysis });
  } catch (err) {
    console.error("analyze error:", err);
    res.status(502).json({ error: "The analysis service didn't return a usable result. Please try again." });
  }
});

// ---- Follow-up Q&A, grounded only in the document text ----

const QA_SYSTEM_PROMPT = `You are CLARIFY, answering follow-up questions about a contract using ONLY the contract text provided — never outside knowledge or assumptions about what's "typical".

Respond with ONLY a single JSON object (no markdown fences):
{ "answer": string, "status": "Answered from contract" | "Not addressed in contract", "evidence": string }

If the contract doesn't cover the question, say so plainly in "answer" and set status to "Not addressed in contract" with an empty "evidence".`;

app.post("/api/sessions/:id/ask", requireSession, async (req, res) => {
  const { question } = req.body || {};
  const text = req.session.text;
  if (!question || !question.trim()) return res.status(400).json({ error: "No question provided." });
  if (!text || !text.trim()) return res.status(400).json({ error: "No document text in this session." });

  try {
    const msg = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 800,
      system: QA_SYSTEM_PROMPT,
      messages: [{ role: "user", content: `Contract text:\n\n${text}\n\nQuestion: ${question}` }],
    });
    const raw = msg.content.map(b => (b.type === "text" ? b.text : "")).join("");
    const answer = parseJsonLoose(raw);
    res.json({ answer });
  } catch (err) {
    console.error("ask error:", err);
    res.status(502).json({ error: "Couldn't get an answer just now." });
  }
});

function parseJsonLoose(raw) {
  const cleaned = raw.trim().replace(/^```json\s*|^```\s*|```$/g, "");
  return JSON.parse(cleaned);
}

app.get("/health", (req, res) => res.json({ ok: true }));

app.listen(PORT, () => console.log(`CLARIFY backend listening on port ${PORT}`));

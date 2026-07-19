import express from "express";
import Anthropic from "@anthropic-ai/sdk";
import { OpenAI } from "openai";
import { GoogleGenAI } from "@google/genai";
import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const MAX_TOKENS = 200;
const PORT = process.env.PORT || 3000;

const SYSTEM_PROMPT = `You are a Senior Software Developer.
You can only answer questions from the field of Software and Computer Science.
Answer in maximum ${MAX_TOKENS} tokens`;

const combinedSystemPrompt = `You are a Senior Software Developer.
Your role is to analyze responses from different LLMs to a user asked question.
Each LLM Response begins with the LLM Name Response Start and ends with a Response Ends statement.
You will be provided with the original user question asked, and the LLM responses.
You will analyze all LLM responses and come up with the best response after the analysis.
Your answer will be limited to the responses provided by the LLMs.
Do not add any new information beyond what is provided in the LLM responses.
Your response should start with Judge LLM Response Start: and end with Judge LLM Response Ends`;

const claudeClient = new Anthropic();
const openaiClient = new OpenAI();
const geminiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const app = express();
app.use(express.json({ limit: "100kb" }));

// Serve the chat frontend (index.html) from the project root
app.use(express.static(__dirname));

app.post("/api/ask", async (req, res) => {
  const { question, history = [] } = req.body ?? {};

  if (!question || typeof question !== "string" || !question.trim()) {
    return res.status(400).json({ error: "Missing 'question' in request body." });
  }

  const cleanHistory = history
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
    .slice(-10);

  const messages = [...cleanHistory, { role: "user", content: question.trim() }];

  const geminiMessages = messages.map((m) => ({
    role: m.role === "assistant" ? "model" : "user",
    parts: [{ text: m.content }],
  }));

  try {
    const [claudeResult, openaiResult, geminiResult] = await Promise.all([
      claudeClient.messages.create({
        model: "claude-opus-4-8",
        max_tokens: MAX_TOKENS,
        messages,
        system: SYSTEM_PROMPT,
      }),
      openaiClient.responses.create({
        model: "gpt-4o-mini",
        input: messages,
        instructions: SYSTEM_PROMPT,
      }),
      geminiClient.models.generateContent({
        model: "gemini-flash-latest",
        contents: geminiMessages,
        config: { systemInstruction: SYSTEM_PROMPT },
      }),
    ]);

    let claudeResponse = "";
    for (const block of claudeResult.content) {
      if (block.type === "text") claudeResponse = block.text;
    }
    const openaiResponse = openaiResult.output_text;
    const geminiResponse = geminiResult.text;

    const combinedResponse = `
User Question: ${question}

LLM1 Response Start:
${claudeResponse}
LLM1 Response End.

LLM2 Response Start:
${openaiResponse}
LLM2 Response End.

LLM3 Response Start:
${geminiResponse}
LLM3 Response End.`;

    const judgeResult = await openaiClient.responses.create({
      model: "o4-mini-2025-04-16",
      input: [...cleanHistory, { role: "user", content: combinedResponse }],
      instructions: combinedSystemPrompt,
    });

    res.json({
      answer: judgeResult.output_text,
      responses: {
        claude: claudeResponse,
        openai: openaiResponse,
        gemini: geminiResponse,
      },
    });
  } catch (e) {
    console.error("judgellm error:", e);
    res.status(500).json({ error: e.message || "Something went wrong." });
  }
});

app.listen(PORT, () => {
  console.log(`JudgeLLM running at http://localhost:${PORT}`);
});
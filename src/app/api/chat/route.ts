import { NextResponse } from "next/server";
import {
  isOllamaAnswerModel,
  parseAnswerModel,
  toLiteLlmModel,
} from "@/lib/answer-model";
import {
  SYSTEM_PROMPT,
  createGeminiClient,
  getGeminiConfig,
  toGeminiContents,
  type ChatApiMessage,
} from "@/lib/gemini";
import { ollamaChat } from "@/lib/ollama";
import { getPineconeConfig } from "@/lib/pinecone";
import {
  buildNoMatchReply,
  formatAggregateAnswer,
  formatListingAnswer,
  formatRateAnswer,
  isAggregateQuestion,
  isListingsMarketQuestion,
  isRateQuestion,
  retrieveListings,
} from "@/lib/rag";
import { parseRetrievalMode, parsePandasEngine, type RetrievalMode, type PandasEngine } from "@/lib/retrieval-mode";
import { addSessionUsage, getSessionUsage } from "@/lib/token-budget";
import { queryLangPanda } from "@/lib/langpanda";
import { chatDocument } from "@/lib/vectorless-docs";
import { retrieveListingsVectorless } from "@/lib/vectorless";
import { parseWordLimit } from "@/lib/word-limit";
import { ReplayableNodeStream } from "next/dist/server/app-render/app-render-prerender-utils";

export const runtime = "nodejs";
export const maxDuration = 300;

type ChatRequestBody = {
  messages?: ChatApiMessage[];
  sessionId?: string;
  mode?: RetrievalMode;
  docId?: string | null;
  model?: string | null;
  pandasEngine?: PandasEngine | null;
  wordLimit?: number | null;
};

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as ChatRequestBody;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const sessionId =
      typeof body.sessionId === "string" && body.sessionId.trim()
        ? body.sessionId.trim()
        : "anonymous";
    const mode = parseRetrievalMode(body.mode);
    const docId =
      typeof body.docId === "string" && body.docId.trim()
        ? body.docId.trim()
        : null;
    const answerModel = parseAnswerModel(body.model);
    const pandasEngine = parsePandasEngine(body.pandasEngine);
    const wordLimit = parseWordLimit(body.wordLimit);
    const useOllama = isOllamaAnswerModel(answerModel);

    if (messages.length === 0) {
      return NextResponse.json(
        { error: "Send at least one message." },
        { status: 400 },
      );
    }

    const config = getGeminiConfig(useOllama ? undefined : answerModel);
    if (
      !useOllama &&
      mode !== "pandas" &&
      !config.apiKey
    ) {
      return NextResponse.json(
        {
          error:
            "GEMINI_API_KEY is missing. Add it to .env.local in the project root, then restart npm run dev.",
        },
        { status: 500 },
      );
    }
    if (mode === "ask_dgb_sup") {
      // ASK DGB-SUP always uses Gemini knowledge (ignores Ollama dropdown).
      const geminiConfig = getGeminiConfig(undefined);
      if (!geminiConfig.apiKey) {
        return NextResponse.json(
          {
            error:
              "ASK DGB-SUP needs GEMINI_API_KEY in .env.local (Gemini market knowledge, no CSV).",
          },
          { status: 500 },
        );
      }
    }

    if (mode === "rag") {
      const pinecone = getPineconeConfig();
      if (!pinecone.apiKey) {
        return NextResponse.json(
          {
            error:
              "PINECONE_API_KEY is missing. Add it to .env.local, then restart npm run dev. Or switch to Vectorless / Pandas mode.",
          },
          { status: 500 },
        );
      }
    }

    const used = getSessionUsage(sessionId);
    const remaining = Math.max(0, config.dailyTokenBudget - used);
    if (remaining <= 0) {
      return NextResponse.json(
        {
          error:
            "Daily token budget reached. Try again tomorrow or raise DAILY_TOKEN_BUDGET.",
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            sessionUsed: used,
            remaining: 0,
            dailyBudget: config.dailyTokenBudget,
          },
          model: answerModel,
        },
        { status: 429 },
      );
    }

    const latestUser = [...messages]
      .reverse()
      .find((message) => message.role === "user");
    const question = latestUser?.content?.trim() || "";

    let retrievedCount = 0;
    const systemInstruction = SYSTEM_PROMPT;

    if (question) {
      // ASK DGB-SUP: Gemini own knowledge only — never CSV, never DDGS/web.
      if (mode === "ask_dgb_sup") {
        try {
          const result = await queryLangPanda(
            question,
            "pandas_gemini",
            answerModel,
            wordLimit,
          );
          const sessionUsed = addSessionUsage(sessionId, 0);
          return NextResponse.json({
            reply: result.reply,
            retrievedCount: 0,
            mode,
            model: answerModel,
            source: result.source || "gemini-only",
            usage: {
              promptTokens: 0,
              completionTokens: 0,
              totalTokens: 0,
              sessionUsed,
              remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
              dailyBudget: config.dailyTokenBudget,
            },
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "ASK DGB-SUP failed";
          return NextResponse.json(
            {
              error: `${message}\n\nTip: run \`npm run langpanda:service\` and ensure GEMINI_API_KEY is set.`,
              model: answerModel,
              mode,
            },
            { status: 500 },
          );
        }
      }

      // LangPanda: exact Pandas code execution on the CSV.
      if (mode === "pandas") {
        try {
          const result = await queryLangPanda(
            question,
            pandasEngine === "pandas_llm" ? "pandas_llm" : "pandas_only",
            answerModel,
          );
          const sessionUsed = addSessionUsage(sessionId, 0);
          return NextResponse.json({
            reply: result.reply,
            retrievedCount: 1,
            mode,
            model: answerModel,
            pandasEngine,
            source: result.source,
            usage: {
              promptTokens: 0,
              completionTokens: 0,
              totalTokens: 0,
              sessionUsed,
              remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
              dailyBudget: config.dailyTokenBudget,
            },
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "LangPanda failed";
          return NextResponse.json(
            {
              error: `${message}\n\nTip: run \`npm run langpanda:service\` in another terminal (port 8770).`,
              model: answerModel,
              mode,
              pandasEngine,
            },
            { status: 500 },
          );
        }
      }

      if (isRateQuestion(question)) {
        const sessionUsed = addSessionUsage(sessionId, 0);
        return NextResponse.json({
          reply: formatRateAnswer(question),
          retrievedCount: 0,
          mode,
          model: answerModel,
          source: "csv-rate",
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            sessionUsed,
            remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
            dailyBudget: config.dailyTokenBudget,
          },
        });
      }

      // PageIndex PDF chat invents sectors/BHK on local models. For listing
      // market questions (sector/BHK/most expensive/etc.), use exact CSV instead.
      if (
        mode === "vectorless" &&
        docId &&
        !isListingsMarketQuestion(question)
      ) {
        try {
          const reply = await chatDocument(
            question,
            docId,
            toLiteLlmModel(answerModel),
          );
          const sessionUsed = addSessionUsage(sessionId, 0);
          return NextResponse.json({
            reply,
            retrievedCount: 1,
            mode,
            docId,
            model: answerModel,
            source: "pageindex",
            usage: {
              promptTokens: 0,
              completionTokens: 0,
              totalTokens: 0,
              sessionUsed,
              remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
              dailyBudget: config.dailyTokenBudget,
            },
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Document chat failed";
          return NextResponse.json(
            { error: message, model: answerModel },
            { status: 500 },
          );
        }
      }

      if (isAggregateQuestion(question)) {
        const sessionUsed = addSessionUsage(sessionId, 0);
        return NextResponse.json({
          reply: formatAggregateAnswer(question),
          retrievedCount: 0,
          mode,
          model: answerModel,
          source: "csv-aggregate",
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            sessionUsed,
            remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
            dailyBudget: config.dailyTokenBudget,
          },
        });
      }

      const pinecone = getPineconeConfig();
      let listings;
      let filtersOverride: Parameters<typeof formatListingAnswer>[2];

      if (mode === "vectorless") {
        const result = await retrieveListingsVectorless(question, answerModel);
        listings = result.listings;
        filtersOverride = result.filters;
      } else {
        listings = await retrieveListings(pinecone.apiKey!, question);
      }
      retrievedCount = listings.length;

      if (listings.length === 0) {
        const sessionUsed = addSessionUsage(sessionId, 0);
        return NextResponse.json({
          reply: buildNoMatchReply(question),
          retrievedCount: 0,
          mode,
          model: answerModel,
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            sessionUsed,
            remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
            dailyBudget: config.dailyTokenBudget,
          },
        });
      }

      const sessionUsed = addSessionUsage(sessionId, 0);
      return NextResponse.json({
        reply: formatListingAnswer(listings, question, filtersOverride),
        retrievedCount,
        mode,
        model: answerModel,
        usage: {
          promptTokens: 0,
          completionTokens: 0,
          totalTokens: 0,
          sessionUsed,
          remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
          dailyBudget: config.dailyTokenBudget,
        },
      });
    }

    if (useOllama) {
      const reply = await ollamaChat({
        model: answerModel,
        messages,
        system: systemInstruction,
        temperature: 0.2,
      });
      const sessionUsed = addSessionUsage(sessionId, 0);
      return NextResponse.json({
        reply,
        retrievedCount,
        mode,
        model: answerModel,
        usage: {
          promptTokens: 0,
          completionTokens: 0,
          totalTokens: 0,
          sessionUsed,
          remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
          dailyBudget: config.dailyTokenBudget,
        },
      });
    }

    const contents = toGeminiContents(messages, config.maxHistoryMessages);
    if (contents.length === 0) {
      return NextResponse.json(
        { error: "Message content was empty." },
        { status: 400 },
      );
    }

    const genAI = createGeminiClient(config.apiKey!);
    const model = genAI.getGenerativeModel({
      model: config.model,
      systemInstruction,
      generationConfig: {
        maxOutputTokens: config.maxOutputTokens,
        temperature: 0.2,
      },
    });

    const result = await model.generateContent({ contents });
    const candidate = result.response.candidates?.[0];
    const finishReason = candidate?.finishReason;
    const reply = result.response.text()?.trim();
    if (!reply) {
      return NextResponse.json(
        { error: "Gemini returned an empty response." },
        { status: 502 },
      );
    }
    if (finishReason && finishReason !== "STOP") {
      console.warn("[api/chat] unusual finishReason:", finishReason);
    }

    const meta = result.response.usageMetadata;
    const promptTokens = meta?.promptTokenCount ?? 0;
    const completionTokens = meta?.candidatesTokenCount ?? 0;
    const totalTokens =
      meta?.totalTokenCount ?? promptTokens + completionTokens;

    const sessionUsed = addSessionUsage(sessionId, totalTokens);

    return NextResponse.json({
      reply,
      retrievedCount,
      mode,
      model: answerModel,
      usage: {
        promptTokens,
        completionTokens,
        totalTokens,
        sessionUsed,
        remaining: Math.max(0, config.dailyTokenBudget - sessionUsed),
        dailyBudget: config.dailyTokenBudget,
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unexpected chat error";
    console.error("[api/chat]", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

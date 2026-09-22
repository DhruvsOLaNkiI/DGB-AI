import type { RetrievalMode } from "@/lib/retrieval-mode";
import type { ChatMessage } from "@/lib/types";

export type ChatUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  sessionUsed: number;
  remaining: number;
  dailyBudget: number;
};

export type ChatApiResult =
  | {
      ok: true;
      reply: string;
      usage: ChatUsage;
      mode?: RetrievalMode;
      model?: string;
      pandasEngine?: "pandas_only" | "pandas_llm";
      source?: string;
    }
  | {
      ok: false;
      error: string;
      usage?: ChatUsage;
      mode?: RetrievalMode;
      model?: string;
      pandasEngine?: "pandas_only" | "pandas_llm";
    };

export async function sendChatRequest(input: {
  messages: ChatMessage[];
  sessionId: string;
  mode?: RetrievalMode;
  docId?: string | null;
  model?: string;
  pandasEngine?: "pandas_only" | "pandas_llm";
  wordLimit?: number;
}): Promise<ChatApiResult> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: input.sessionId,
      mode: input.mode ?? "rag",
      docId: input.docId ?? null,
      model: input.model,
      pandasEngine: input.pandasEngine ?? "pandas_only",
      wordLimit: input.wordLimit,
      messages: input.messages.map((m) => ({
        role: m.role,
        content: m.content,
        imageDataUrl: m.imageDataUrl,
      })),
    }),
  });

  const data = (await res.json()) as {
    reply?: string;
    error?: string;
    usage?: ChatUsage;
    mode?: RetrievalMode;
    model?: string;
    pandasEngine?: "pandas_only" | "pandas_llm";
    source?: string;
  };

  if (!res.ok || !data.reply) {
    return {
      ok: false,
      error: data.error || `Request failed (${res.status})`,
      usage: data.usage,
      mode: data.mode,
      model: data.model,
      pandasEngine: data.pandasEngine,
    };
  }

  return {
    ok: true,
    reply: data.reply,
    usage: data.usage!,
    mode: data.mode,
    model: data.model,
    pandasEngine: data.pandasEngine,
    source: data.source,
  };
}

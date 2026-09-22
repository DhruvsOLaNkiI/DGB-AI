export type RetrievalMode = "rag" | "vectorless" | "pandas" | "ask_dgb_sup";

/**
 * When mode is pandas:
 * - pandas_only → exact CSV filters only
 * - pandas_llm  → CSV first, then UI LLM + web fallback
 * ASK DGB-SUP is a separate top-level mode (Gemini own knowledge only —
 * never CSV, never DuckDuckGo / web search).
 */
export type PandasEngine = "pandas_only" | "pandas_llm";

export const RETRIEVAL_MODE_KEY = "dbg-ai-retrieval-mode";
export const PANDAS_ENGINE_KEY = "dbg-ai-pandas-engine";

export function isRetrievalMode(value: unknown): value is RetrievalMode {
  return (
    value === "rag" ||
    value === "vectorless" ||
    value === "pandas" ||
    value === "ask_dgb_sup"
  );
}

export function parseRetrievalMode(value: unknown): RetrievalMode {
  // Legacy: Gemini-under-Pandas → ASK DGB-SUP top-level mode
  if (value === "pandas_gemini" || value === "gemini_only") {
    return "ask_dgb_sup";
  }
  return isRetrievalMode(value) ? value : "rag";
}

export function isPandasEngine(value: unknown): value is PandasEngine {
  return value === "pandas_only" || value === "pandas_llm";
}

export function parsePandasEngine(value: unknown): PandasEngine {
  // Legacy gemini engine moved to ASK DGB-SUP mode
  if (value === "pandas_gemini") return "pandas_only";
  return isPandasEngine(value) ? value : "pandas_only";
}

export function retrievalModeLabel(mode: RetrievalMode): string {
  if (mode === "vectorless") return "Vectorless";
  if (mode === "pandas") return "Pandas";
  if (mode === "ask_dgb_sup") return "ASK DGB-SUP";
  return "Vector RAG";
}

export function pandasEngineLabel(engine: PandasEngine): string {
  return engine === "pandas_llm" ? "Pandas + LLM" : "Only Pandas";
}

/** Human label for which path delivered the answer. */
export function pandasDeliveryLabel(source: string | undefined | null): string {
  switch ((source || "").trim()) {
    case "pandas-only":
      return "Only Pandas · CSV";
    case "pandas-llm":
      return "Pandas + LLM · CSV";
    case "web-fallback":
      return "Pandas + LLM · Web";
    case "web-fallback-gemini":
    case "gemini-knowledge":
      return "Pandas + LLM · Gemini";
    case "pandas-gemini":
    case "gemini-only":
      return "ASK DGB-SUP · Gemini only";
    case "pandas-empty":
      return "Only Pandas · No match";
    case "out-of-scope":
      return "Out of scope";
    case "error":
      return "Error";
    default:
      return source ? String(source) : "";
  }
}

/** Models the UI can pick for LLM answers (Gemini + local Ollama). */

export type AnswerModelOption = {
  id: string;
  label: string;
  provider: "gemini" | "ollama";
  /** LiteLLM / PageIndex id when different from UI id */
  litellmId?: string;
};

export const ANSWER_MODEL_KEY = "dbg-ai-answer-model";

/** Curated answer models. Embeddings stay on MiniLM. */
export const ANSWER_MODELS: AnswerModelOption[] = [
  { id: "gemini-3.6-flash", label: "Gemini 3.6 Flash", provider: "gemini" },
  { id: "gemini-flash-latest", label: "Gemini Flash (latest)", provider: "gemini" },
  { id: "gemini-3.5-flash-lite", label: "Gemini 3.5 Flash-Lite", provider: "gemini" },
  { id: "gemini-3.5-flash", label: "Gemini 3.5 Flash", provider: "gemini" },
  {
    id: "ollama/llama3.2:3b",
    label: "Ollama (llama3.2:3b)",
    provider: "ollama",
    litellmId: "ollama/llama3.2:3b",
  },
];

const ALLOWED = new Set(ANSWER_MODELS.map((m) => m.id));

export function defaultAnswerModel(): string {
  return (
    process.env.GEMINI_MODEL?.trim() ||
    ANSWER_MODELS[0]?.id ||
    "gemini-3.6-flash"
  );
}

export function isAnswerModel(value: unknown): value is string {
  return typeof value === "string" && ALLOWED.has(value.trim());
}

export function isOllamaAnswerModel(id: string): boolean {
  const option = ANSWER_MODELS.find((m) => m.id === id);
  return option?.provider === "ollama" || id.startsWith("ollama/");
}

/** Resolve a client-requested model to a safe allowlisted id. */
/** Old UI ids → models that new Gemini API keys can still call. */
const LEGACY_ANSWER_MODELS: Record<string, string> = {
  "gemini-2.5-flash": "gemini-3.6-flash",
  "gemini-2.5-pro": "gemini-3.6-flash",
  "gemini-2.0-flash": "gemini-3.6-flash",
  "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
};

export function parseAnswerModel(value: unknown): string {
  if (typeof value === "string") {
    const id = value.trim();
    if (ALLOWED.has(id)) return id;
    const mapped = LEGACY_ANSWER_MODELS[id];
    if (mapped && ALLOWED.has(mapped)) return mapped;
    // Allow the env default even if not in the static list (custom deploy).
    const envDefault = process.env.GEMINI_MODEL?.trim();
    if (envDefault && id === envDefault) return id;
    if (id.startsWith("ollama/")) return id;
  }
  return defaultAnswerModel();
}

export function answerModelLabel(id: string): string {
  return ANSWER_MODELS.find((m) => m.id === id)?.label ?? id;
}

/** LiteLLM / PageIndex model id (gemini/… or ollama/…). */
export function toLiteLlmModel(id: string): string {
  const option = ANSWER_MODELS.find((m) => m.id === id);
  if (option?.litellmId) return option.litellmId;
  if (id.startsWith("ollama/") || id.startsWith("gemini/") || id.startsWith("groq/")) {
    return id;
  }
  if (isOllamaAnswerModel(id)) {
    return id.startsWith("ollama/") ? id : `ollama/${id}`;
  }
  const bare = id.replace(/^gemini\//, "").trim();
  return `gemini/${bare || defaultAnswerModel()}`;
}

/** @deprecated use toLiteLlmModel */
export function toLiteLlmGeminiModel(id: string): string {
  return toLiteLlmModel(id);
}

/** Bare Ollama model name for the local HTTP API (no ollama/ prefix). */
export function toOllamaModelName(id: string): string {
  const litellm = toLiteLlmModel(id);
  return litellm.replace(/^ollama\//, "").trim() || "llama3.2:3b";
}

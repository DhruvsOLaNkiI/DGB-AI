import type { ChatApiMessage } from "@/lib/gemini";
import { toOllamaModelName } from "@/lib/answer-model";

function ollamaBaseUrl(): string {
  return (
    process.env.OLLAMA_API_BASE?.trim() ||
    process.env.OLLAMA_HOST?.trim() ||
    "http://127.0.0.1:11434"
  ).replace(/\/$/, "");
}

export async function ollamaGenerate(input: {
  model: string;
  prompt: string;
  system?: string;
  temperature?: number;
}): Promise<string> {
  const model = toOllamaModelName(input.model);
  const res = await fetch(`${ollamaBaseUrl()}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt: input.prompt,
      system: input.system,
      stream: false,
      options: {
        temperature: input.temperature ?? 0.2,
      },
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `Ollama error (${res.status}): ${body.trim() || res.statusText}. Is Ollama running at ${ollamaBaseUrl()} with model "${model}"?`,
    );
  }

  const data = (await res.json()) as { response?: string };
  const text = data.response?.trim();
  if (!text) {
    throw new Error("Ollama returned an empty response.");
  }
  return text;
}

export async function ollamaChat(input: {
  model: string;
  messages: ChatApiMessage[];
  system?: string;
  temperature?: number;
}): Promise<string> {
  const model = toOllamaModelName(input.model);
  const messages: Array<{ role: string; content: string }> = [];
  if (input.system?.trim()) {
    messages.push({ role: "system", content: input.system.trim() });
  }
  for (const m of input.messages) {
    const content = m.content?.trim();
    if (!content && !m.imageDataUrl) continue;
    // Local Ollama vision varies by model — send text hint if image present.
    const text =
      content ||
      (m.imageDataUrl
        ? "Please review this property photo and share useful observations."
        : "");
    if (!text) continue;
    messages.push({
      role: m.role === "assistant" ? "assistant" : "user",
      content: text,
    });
  }

  const res = await fetch(`${ollamaBaseUrl()}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: {
        temperature: input.temperature ?? 0.2,
      },
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `Ollama error (${res.status}): ${body.trim() || res.statusText}. Is Ollama running at ${ollamaBaseUrl()} with model "${model}"?`,
    );
  }

  const data = (await res.json()) as {
    message?: { content?: string };
  };
  const text = data.message?.content?.trim();
  if (!text) {
    throw new Error("Ollama returned an empty response.");
  }
  return text;
}

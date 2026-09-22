import { spawn } from "node:child_process";
import path from "node:path";
import type { PandasEngine } from "@/lib/retrieval-mode";

/** Internal engine: pandas_gemini = ASK DGB-SUP (Gemini own knowledge; no CSV / no web). */
type LangPandaEngine = PandasEngine | "pandas_gemini";

const ROOT = process.cwd();
const LANGPANDA = path.join(ROOT, "langpanda");
const PYTHON = path.join(LANGPANDA, ".venv", "bin", "python");

const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]|\[\d+(?:;\d+)*m/g;

/** Strip AgentExecutor / ANSI noise if Python stdout ever mixes logs. */
function cleanPandasReply(raw: string): string {
  let text = raw.replace(ANSI_RE, "").replace(/\r/g, "").trim();
  const finalMatch = text.match(
    /Final\s*Answer\s*:\s*([\s\S]+?)(?:\n\s*(?:Thought|Action|>\s*Finished)|$)/i,
  );
  if (finalMatch?.[1]) text = finalMatch[1].trim();
  text = text
    .split("\n")
    .filter(
      (line) =>
        !/>\s*Entering new AgentExecutor chain/i.test(line) &&
        !/>\s*Finished chain/i.test(line) &&
        !/^\s*(Thought|Action|Action Input|Observation)\s*:/i.test(line),
    )
    .join("\n")
    .trim();
  return text || raw.trim();
}

function serviceUrl(): string {
  return (
    process.env.LANGPANDA_SERVICE_URL?.trim() || "http://127.0.0.1:8770"
  ).replace(/\/$/, "");
}

export function langpandaRoot(): string {
  return LANGPANDA;
}

async function askViaHttp(
  question: string,
  engine: LangPandaEngine,
  model?: string,
  wordLimit?: number,
): Promise<{ reply: string; source?: string } | null> {
  const url = `${serviceUrl()}/pandas-chat`;
  const useLlm = engine === "pandas_llm" || engine === "pandas_gemini";
  const resolvedModel =
    engine === "pandas_gemini"
      ? forceGeminiModelId(model)
      : model || null;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        use_llm: useLlm,
        model: resolvedModel,
        force_gemini: engine === "pandas_gemini",
        word_limit:
          engine === "pandas_gemini" && typeof wordLimit === "number"
            ? wordLimit
            : null,
      }),
      signal: AbortSignal.timeout(120_000),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `LangPanda HTTP ${res.status}`);
    }
    const data = (await res.json()) as { reply?: string; source?: string };
    if (!data.reply?.trim()) return null;
    return {
      reply: cleanPandasReply(data.reply),
      source: data.source,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/ECONNREFUSED|fetch failed|AbortError|timeout/i.test(message)) {
      return null;
    }
    throw error;
  }
}

/** Always a Gemini model id when Gemini Only is selected. */
function forceGeminiModelId(model?: string): string {
  const raw = (model || "").trim();
  if (raw && /gemini/i.test(raw) && !/^ollama\//i.test(raw)) {
    return raw.startsWith("gemini/") ? raw.slice("gemini/".length) : raw;
  }
  return (
    process.env.LANGPANDA_GEMINI_MODEL?.trim() ||
    process.env.GEMINI_MODEL?.trim() ||
    "gemini-3.6-flash"
  );
}

function askViaPython(
  question: string,
  engine: LangPandaEngine,
  model?: string,
  wordLimit?: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const useLlm = engine === "pandas_llm" || engine === "pandas_gemini";
    const resolvedModel =
      engine === "pandas_gemini" ? forceGeminiModelId(model) : model || "";
    const limit =
      engine === "pandas_gemini" && typeof wordLimit === "number"
        ? String(wordLimit)
        : "0";
    const child = spawn(
      PYTHON,
      [
        "-c",
        // Marker keeps final answer separable if verbose ever leaks to stdout.
        "from agent import query_real_estate_agent; import sys; "
          + "wl=int(sys.argv[5]) if sys.argv[5].isdigit() else None; "
          + "reply = query_real_estate_agent(sys.argv[1], use_llm=(sys.argv[2]=='1'), model=(sys.argv[3] or None), force_gemini=(sys.argv[4]=='1'), word_limit=wl); "
          + "sys.stdout.write('__LANGPANDA_REPLY__\\n' + reply + '\\n')",
        question,
        useLlm ? "1" : "0",
        resolvedModel,
        engine === "pandas_gemini" ? "1" : "0",
        limit,
      ],
      {
        cwd: LANGPANDA,
        env: process.env,
      },
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("LangPanda Python timed out after 120s"));
    }, 120_000);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(
        new Error(
          `${err.message}. Start the service: npm run langpanda:service`,
        ),
      );
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      const marker = "__LANGPANDA_REPLY__\n";
      const marked = stdout.includes(marker)
        ? stdout.slice(stdout.lastIndexOf(marker) + marker.length)
        : stdout;
      const reply = cleanPandasReply(marked);
      if (code !== 0 || !reply) {
        reject(
          new Error(
            stderr.trim() ||
              reply ||
              `LangPanda exited with code ${code}. Run: npm run langpanda:service`,
          ),
        );
        return;
      }
      if (reply.startsWith("Execution error:")) {
        reject(new Error(reply));
        return;
      }
      resolve(reply);
    });
  });
}

/** Ask LangPanda (HTTP service, else local Python). */
export async function queryLangPanda(
  question: string,
  engine: LangPandaEngine = "pandas_only",
  model?: string,
  wordLimit?: number,
): Promise<{ reply: string; source: string }> {
  const viaHttp = await askViaHttp(question, engine, model, wordLimit);
  if (viaHttp?.reply) {
    return {
      reply: viaHttp.reply,
      source:
        viaHttp.source ||
        (engine === "pandas_gemini"
          ? "gemini-only"
          : engine === "pandas_llm"
            ? "pandas-llm"
            : "pandas-only"),
    };
  }
  const reply = await askViaPython(question, engine, model, wordLimit);
  return {
    reply,
    source:
      engine === "pandas_gemini"
        ? "gemini-only"
        : engine === "pandas_llm"
          ? "pandas-llm"
          : "pandas-only",
  };
}

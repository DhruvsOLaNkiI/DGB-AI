import { spawn } from "node:child_process";
import path from "node:path";
import type { PandasEngine } from "@/lib/retrieval-mode";
import type { AskProfile } from "@/lib/ask-profile";

/** Internal engine: pandas_gemini = ASK DGB-SUP (Gemini own knowledge; no CSV / no web). */
type LangPandaEngine = PandasEngine | "pandas_gemini";

export type LangPandaHistoryTurn = {
  role: "user" | "assistant";
  content: string;
};

export type AskDgbContext = {
  history: LangPandaHistoryTurn[];
  olderSummary?: string;
};

const ROOT = process.cwd();
const LANGPANDA = path.join(ROOT, "langpanda");
const PYTHON = path.join(LANGPANDA, ".venv", "bin", "python");

/** Max prior turns sent to ASK DGB-SUP (user+assistant messages). */
export const ASK_DGB_HISTORY_MAX_MESSAGES = 8;
/** Max older user asks kept as compressed summary beyond the buffer. */
export const ASK_DGB_OLDER_USER_MAX = 6;
export const ASK_DGB_OLDER_USER_WORDS = 40;

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

function truncateWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) return text.trim();
  return `${words.slice(0, maxWords).join(" ")}…`;
}

function collectAskTurns(
  messages: Array<{ role: string; content?: string }>,
  currentQuestion: string,
): LangPandaHistoryTurn[] {
  const q = currentQuestion.trim();
  const turns: LangPandaHistoryTurn[] = [];
  for (const m of messages) {
    const role =
      m.role === "assistant" ? "assistant" : m.role === "user" ? "user" : null;
    const content = (m.content || "").trim();
    if (!role || !content) continue;
    turns.push({ role, content });
  }
  // Drop trailing duplicate of the current question.
  if (
    turns.length &&
    turns[turns.length - 1]!.role === "user" &&
    turns[turns.length - 1]!.content === q
  ) {
    turns.pop();
  }
  return turns;
}

/**
 * Phase 1+3: recent buffer (last N) + compressed older user asks
 * so long chats stay within the token budget.
 */
export function buildAskDgbContext(
  messages: Array<{ role: string; content?: string }>,
  currentQuestion: string,
): AskDgbContext {
  const turns = collectAskTurns(messages, currentQuestion);
  if (!turns.length) return { history: [] };

  const older = turns.slice(0, -ASK_DGB_HISTORY_MAX_MESSAGES);
  const recent = turns.slice(-ASK_DGB_HISTORY_MAX_MESSAGES);
  const olderUsers = older
    .filter((t) => t.role === "user")
    .slice(-ASK_DGB_OLDER_USER_MAX)
    .map((t) => truncateWords(t.content, ASK_DGB_OLDER_USER_WORDS));

  const olderSummary = olderUsers.length
    ? olderUsers.map((t, i) => `${i + 1}. ${t}`).join("\n")
    : undefined;

  return { history: recent, olderSummary };
}

/** Build ASK DGB-SUP history from chat messages (excludes empty; keeps last N). */
export function buildAskDgbHistory(
  messages: Array<{ role: string; content?: string }>,
  currentQuestion: string,
): LangPandaHistoryTurn[] {
  return buildAskDgbContext(messages, currentQuestion).history;
}

async function askViaHttp(
  question: string,
  engine: LangPandaEngine,
  model?: string,
  wordLimit?: number,
  history?: LangPandaHistoryTurn[],
  olderSummary?: string,
  profile?: AskProfile | null,
): Promise<{ reply: string; source?: string; askState?: AskProfile | null } | null> {
  const url = `${serviceUrl()}/pandas-chat`;
  const useLlm =
    engine === "pandas_llm" ||
    engine === "pandas_gemini" ||
    engine === "firecrawl_llm";
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
        force_firecrawl: engine === "firecrawl_llm",
        word_limit:
          engine === "pandas_gemini" && typeof wordLimit === "number"
            ? wordLimit
            : null,
        history:
          engine === "pandas_gemini" && history && history.length > 0
            ? history
            : null,
        older_summary:
          engine === "pandas_gemini" && olderSummary?.trim()
            ? olderSummary.trim()
            : null,
        profile: engine === "pandas_gemini" && profile ? profile : null,
      }),
      signal: AbortSignal.timeout(180_000),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `LangPanda HTTP ${res.status}`);
    }
    const data = (await res.json()) as {
      reply?: string;
      source?: string;
      ask_state?: AskProfile | null;
    };
    if (!data.reply?.trim()) return null;
    return {
      reply: cleanPandasReply(data.reply),
      source: data.source,
      askState: data.ask_state ?? null,
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
  history?: LangPandaHistoryTurn[],
  olderSummary?: string,
  profile?: AskProfile | null,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const useLlm =
      engine === "pandas_llm" ||
      engine === "pandas_gemini" ||
      engine === "firecrawl_llm";
    const forceFirecrawl = engine === "firecrawl_llm" ? "1" : "0";
    const resolvedModel =
      engine === "pandas_gemini" ? forceGeminiModelId(model) : model || "";
    const limit =
      engine === "pandas_gemini" && typeof wordLimit === "number"
        ? String(wordLimit)
        : "0";
    const historyJson =
      engine === "pandas_gemini" && history && history.length > 0
        ? JSON.stringify(history)
        : "";
    const olderJson =
      engine === "pandas_gemini" && olderSummary?.trim()
        ? olderSummary.trim()
        : "";
    const profileJson =
      engine === "pandas_gemini" && profile
        ? JSON.stringify(profile)
        : "";
    const child = spawn(
      PYTHON,
      [
        "-c",
        "from agent import query_real_estate_agent; import sys, json, os; "
          + "wl=int(sys.argv[5]) if sys.argv[5].isdigit() else None; "
          + "raw=os.environ.get('LANGPANDA_CHAT_HISTORY') or ''; "
          + "hist=json.loads(raw) if raw.strip() else None; "
          + "older=(os.environ.get('LANGPANDA_OLDER_SUMMARY') or '').strip() or None; "
          + "praw=os.environ.get('LANGPANDA_ASK_PROFILE') or ''; "
          + "prof=json.loads(praw) if praw.strip() else None; "
          + "reply = query_real_estate_agent(sys.argv[1], use_llm=(sys.argv[2]=='1'), model=(sys.argv[3] or None), force_gemini=(sys.argv[4]=='1'), word_limit=wl, history=hist, older_summary=older, profile=prof, force_firecrawl=(sys.argv[6]=='1')); "
          + "sys.stdout.write('__LANGPANDA_REPLY__\\n' + reply + '\\n')",
        question,
        useLlm ? "1" : "0",
        resolvedModel,
        engine === "pandas_gemini" ? "1" : "0",
        limit,
        forceFirecrawl,
      ],
      {
        cwd: LANGPANDA,
        env: {
          ...process.env,
          ...(historyJson ? { LANGPANDA_CHAT_HISTORY: historyJson } : {}),
          ...(olderJson ? { LANGPANDA_OLDER_SUMMARY: olderJson } : {}),
          ...(profileJson ? { LANGPANDA_ASK_PROFILE: profileJson } : {}),
        },
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
  history?: LangPandaHistoryTurn[],
  olderSummary?: string,
  profile?: AskProfile | null,
): Promise<{ reply: string; source: string; askState?: AskProfile | null }> {
  const viaHttp = await askViaHttp(
    question,
    engine,
    model,
    wordLimit,
    history,
    olderSummary,
    profile,
  );
  if (viaHttp?.reply) {
    return {
      reply: viaHttp.reply,
      source:
        viaHttp.source ||
        (engine === "pandas_gemini"
          ? "gemini-only"
          : engine === "firecrawl_llm"
            ? "firecrawl-llm"
            : engine === "pandas_llm"
              ? "pandas-llm"
              : "pandas-only"),
      askState: viaHttp.askState ?? null,
    };
  }
  const reply = await askViaPython(
    question,
    engine,
    model,
    wordLimit,
    history,
    olderSummary,
    profile,
  );
  return {
    reply,
    source:
      engine === "pandas_gemini"
        ? "gemini-only"
        : engine === "firecrawl_llm"
          ? "firecrawl-llm"
          : engine === "pandas_llm"
            ? "pandas-llm"
            : "pandas-only",
    askState: null,
  };
}

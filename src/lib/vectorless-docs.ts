import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const VECTORLESS = path.join(ROOT, "vecrtorless rag");
const PYTHON = path.join(VECTORLESS, ".venv", "bin", "python");
const REGISTRY = path.join(VECTORLESS, "data", "documents.json");
const UPLOADS = path.join(VECTORLESS, "data", "uploads");

export type VectorlessDocument = {
  doc_id: string;
  name: string;
  path?: string;
  mode?: string;
  indexed_at?: string;
};

export function vectorlessPython(): string {
  return PYTHON;
}

export function vectorlessRoot(): string {
  return VECTORLESS;
}

export async function readDocumentRegistry(): Promise<VectorlessDocument[]> {
  try {
    const raw = await readFile(REGISTRY, "utf8");
    const data = JSON.parse(raw) as unknown;
    return Array.isArray(data) ? (data as VectorlessDocument[]) : [];
  } catch {
    return [];
  }
}

export async function ensureUploadsDir(): Promise<string> {
  await mkdir(UPLOADS, { recursive: true });
  return UPLOADS;
}

export function runPython(
  args: string[],
  opts?: { timeoutMs?: number },
): Promise<{ code: number; stdout: string; stderr: string }> {
  const timeoutMs = opts?.timeoutMs ?? 600_000;
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, args, {
      cwd: VECTORLESS,
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Python timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

export async function saveUploadedFile(
  filename: string,
  bytes: Buffer,
): Promise<string> {
  const dir = await ensureUploadsDir();
  const safe = path.basename(filename).replace(/[^\w.\- ()]+/g, "_");
  const lower = safe.toLowerCase();
  const fallback = lower.endsWith(".csv")
    ? `upload-${Date.now()}.csv`
    : `upload-${Date.now()}.pdf`;
  const target = path.join(dir, safe || fallback);
  await writeFile(target, bytes);
  return target;
}

/** @deprecated use saveUploadedFile */
export async function saveUploadedPdf(
  filename: string,
  bytes: Buffer,
): Promise<string> {
  return saveUploadedFile(filename, bytes);
}

export async function indexDocumentFile(
  filePath: string,
): Promise<VectorlessDocument> {
  const result = await runPython(["scripts/index_document.py", filePath]);
  if (result.code !== 0) {
    throw new Error(
      result.stderr.trim() || result.stdout.trim() || "Failed to index document",
    );
  }
  const docs = await readDocumentRegistry();
  const latest = docs[0];
  if (!latest?.doc_id) {
    throw new Error("Indexing finished but no document was registered.");
  }
  return latest;
}

export async function indexPdfFile(pdfPath: string): Promise<VectorlessDocument> {
  return indexDocumentFile(pdfPath);
}

export async function chatDocument(
  question: string,
  docId: string,
  modelOverride?: string,
): Promise<string> {
  const args = ["scripts/chat_document.py", "--doc-id", docId];
  if (modelOverride?.trim()) {
    args.push("--model", modelOverride.trim());
  }
  args.push(question);
  const result = await runPython(args);
  if (result.code !== 0) {
    const raw =
      result.stderr.trim() || result.stdout.trim() || "Document chat failed";
    if (/rate.?limit|429|quota|resource_exhausted/i.test(raw)) {
      throw new Error(
        "Gemini free-tier quota exceeded for document chat. Wait ~1 minute and retry, or set the dropdown to “Listings (CSV tree)” for property count questions (no PageIndex chat needed). To use local models instead, set PAGEINDEX_LLM=ollama in .env.local.",
      );
    }
    throw new Error(raw);
  }
  const lines = result.stdout.trim().split("\n");
  const filtered = lines.filter((line) => !line.startsWith("Using latest doc:"));
  return filtered.join("\n").trim() || result.stdout.trim();
}

"use client";

import { useEffect, useState } from "react";
import type { RetrievalMode } from "@/lib/retrieval-mode";

const DOC_KEY = "dbg-ai-vectorless-doc-id";

type VectorlessDocument = {
  doc_id: string;
  name: string;
  path?: string;
  mode?: string;
  indexed_at?: string;
};

type Props = {
  mode: RetrievalMode;
  docId: string | null;
  onDocIdChange: (docId: string | null) => void;
};

export function VectorlessDocBar({ mode, docId, onDocIdChange }: Props) {
  const [docs, setDocs] = useState<VectorlessDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== "vectorless") return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/vectorless/documents");
        const data = (await res.json()) as { documents?: VectorlessDocument[] };
        if (cancelled) return;
        const list = data.documents ?? [];
        setDocs(list);
        // Restore previously selected PageIndex doc (upload once, reuse).
        const saved = window.localStorage.getItem(DOC_KEY);
        if (!docId && saved && list.some((d) => d.doc_id === saved)) {
          onDocIdChange(saved);
        }
      } catch {
        if (!cancelled) setDocs([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode, docId, onDocIdChange]);

  if (mode !== "vectorless") return null;

  async function handleUpload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/vectorless/documents", {
        method: "POST",
        body: form,
      });
      const data = (await res.json()) as {
        document?: VectorlessDocument;
        error?: string;
      };
      if (!res.ok || !data.document) {
        throw new Error(data.error || "Upload failed");
      }
      setDocs((prev) => [
        data.document!,
        ...prev.filter((d) => d.doc_id !== data.document!.doc_id),
      ]);
      onDocIdChange(data.document.doc_id);
      window.localStorage.setItem(DOC_KEY, data.document.doc_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="mx-3 mb-1 sm:mx-6">
      <div className="flex flex-wrap items-center gap-2 rounded-xl bg-white/90 px-3 py-2 text-[12px] text-ink shadow-sm ring-1 ring-black/5">
        <span className="font-medium text-ink/70">Source</span>
        <select
          className="max-w-[180px] rounded-lg bg-black/5 px-2 py-1 text-[12px] outline-none sm:max-w-[240px]"
          value={docId ?? ""}
          onChange={(e) => {
            const next = e.target.value || null;
            onDocIdChange(next);
            if (next) window.localStorage.setItem(DOC_KEY, next);
            else window.localStorage.removeItem(DOC_KEY);
          }}
          disabled={uploading}
        >
          <option value="">Listings CSV (not PageIndex)</option>
          {docs.map((doc) => (
            <option key={doc.doc_id} value={doc.doc_id}>
              PageIndex: {doc.name}
            </option>
          ))}
        </select>

        <label className="cursor-pointer rounded-lg bg-accent px-2.5 py-1 text-[12px] font-medium text-white hover:opacity-90 disabled:opacity-50">
          {uploading ? "Indexing tree…" : "Upload once"}
          <input
            type="file"
            accept="application/pdf,.pdf,text/csv,.csv"
            className="hidden"
            disabled={uploading}
            onChange={(e) => {
              void handleUpload(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </label>

        <span className="text-ink/45">
          {docId
            ? "Using saved PageIndex tree — ask freely (no re-upload)."
            : "Select a PageIndex doc above, or stay on Listings CSV."}
        </span>
      </div>
      {error && <p className="mt-1 text-xs text-rose-700">{error}</p>}
    </div>
  );
}

export { DOC_KEY };

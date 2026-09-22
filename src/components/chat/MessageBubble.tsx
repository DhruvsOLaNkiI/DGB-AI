"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import type { ChatMessage } from "@/lib/types";
import { pandasDeliveryLabel } from "@/lib/retrieval-mode";

type MessageBubbleProps = {
  message: ChatMessage;
  onQuoteReply?: (text: string) => void;
};

export function MessageBubble({ message, onQuoteReply }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const bodyRef = useRef<HTMLDivElement>(null);
  const [replyUi, setReplyUi] = useState<{
    text: string;
    top: number;
    left: number;
  } | null>(null);

  const clearReply = useCallback(() => setReplyUi(null), []);

  useEffect(() => {
    function onDocMouseDown(e: MouseEvent) {
      const target = e.target as HTMLElement;
      if (target.closest("[data-reply-btn]")) return;
      clearReply();
    }
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [clearReply]);

  function handleMouseUp() {
    if (isUser || !onQuoteReply || !bodyRef.current) return;
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) {
      clearReply();
      return;
    }
    const text = selection.toString().trim();
    if (!text || text.length < 2) {
      clearReply();
      return;
    }
    if (!bodyRef.current.contains(selection.anchorNode)) {
      clearReply();
      return;
    }
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const parentRect = bodyRef.current.getBoundingClientRect();
    setReplyUi({
      text,
      top: rect.top - parentRect.top - 36,
      left: Math.min(
        Math.max(rect.left - parentRect.left + rect.width / 2 - 40, 0),
        parentRect.width - 90,
      ),
    });
  }

  if (isUser) {
    return (
      <div className="flex w-full justify-end">
        <div className="max-w-[min(100%,28rem)] rounded-2xl bg-white/90 px-3.5 py-2 text-[13px] font-normal leading-5 text-ink shadow-sm">
          {message.imageDataUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={message.imageDataUrl}
              alt="Uploaded"
              className="mb-2 max-h-40 w-auto rounded-xl object-cover"
            />
          )}
          {message.content ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <article className="relative w-full max-w-2xl">
      {message.imageDataUrl && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={message.imageDataUrl}
          alt="Uploaded"
          className="mb-3 max-h-48 w-auto rounded-xl object-cover"
        />
      )}

      {(() => {
        const src = (message.deliverySource || "").trim();
        // Hide ASK DGB-SUP “Gemini only” chip — answer stands alone.
        if (src === "gemini-only" || src === "pandas-gemini") return null;
        if (!src && !message.pandasEngine) return null;
        const label = src
          ? pandasDeliveryLabel(src)
          : message.pandasEngine === "pandas_llm"
            ? "Pandas + LLM · CSV"
            : "Only Pandas · CSV";
        return (
          <div className="mb-1.5">
            <span className="inline-flex items-center rounded-full border border-black/10 bg-black/[0.04] px-2 py-0.5 text-[10px] font-medium tracking-wide text-ink/55">
              {label}
            </span>
          </div>
        );
      })()}

      <div ref={bodyRef} className="relative" onMouseUp={handleMouseUp}>
        {replyUi && (
          <button
            type="button"
            data-reply-btn
            className="selection-reply-btn absolute z-20 inline-flex items-center gap-1 rounded-full bg-accent px-2.5 py-1 text-[11px] font-medium text-white shadow-md"
            style={{ top: replyUi.top, left: replyUi.left }}
            onClick={() => {
              onQuoteReply?.(replyUi.text);
              clearReply();
              window.getSelection()?.removeAllRanges();
            }}
          >
            Reply
          </button>
        )}
        <div className="chat-answer m-0 text-ink selection:bg-accent-soft">
          {renderAnswer(message.content)}
        </div>
      </div>
    </article>
  );
}

type Source = { host: string; name: string; url: string };

type AnswerBlock =
  | { type: "table"; headers: string[]; rows: string[][] }
  | { type: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; text: string }
  | { type: "text"; text: string };

function isTableSeparator(line: string): boolean {
  return /^\s*\|?[\s:|-]+\|[\s:|-|]*\|?\s*$/.test(line) && /---/.test(line);
}

function splitTableRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

function parseAnswerBlocks(content: string): AnswerBlock[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: AnswerBlock[] = [];
  let i = 0;
  let textBuf: string[] = [];

  const flushText = () => {
    const text = textBuf.join("\n").trimEnd();
    if (text.trim()) blocks.push({ type: "text", text });
    textBuf = [];
  };

  while (i < lines.length) {
    const line = lines[i] ?? "";
    const heading = /^(#{1,6})\s+(.+)$/.exec(line.trim());
    if (heading) {
      flushText();
      const level = Math.min(6, heading[1]!.length) as 1 | 2 | 3 | 4 | 5 | 6;
      blocks.push({
        type: "heading",
        level,
        text: heading[2]!.trim(),
      });
      i += 1;
      continue;
    }

    // Markdown pipe table: header + separator + rows
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      isTableSeparator(lines[i + 1] ?? "")
    ) {
      flushText();
      const headers = splitTableRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && (lines[i] ?? "").includes("|")) {
        const rowLine = lines[i] ?? "";
        if (!rowLine.trim() || isTableSeparator(rowLine)) break;
        rows.push(splitTableRow(rowLine));
        i += 1;
      }
      if (headers.length && rows.length) {
        blocks.push({ type: "table", headers, rows });
      }
      continue;
    }

    textBuf.push(line);
    i += 1;
  }
  flushText();
  return blocks;
}

function renderAnswer(content: string): ReactNode[] {
  return parseAnswerBlocks(content).map((block, bi) => {
    if (block.type === "table") {
      return (
        <div key={`t-${bi}`} className="chat-compare-wrap my-3 overflow-x-auto">
          <table className="chat-compare-table">
            <thead>
              <tr>
                {block.headers.map((h, hi) => (
                  <th key={hi}>{inlineFormat(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, ri) => (
                <tr key={ri}>
                  {block.headers.map((_, ci) => (
                    <td key={ci}>{inlineFormat(row[ci] ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    if (block.type === "heading") {
      const size =
        block.level <= 2
          ? "mb-1.5 mt-4 text-[14px] font-semibold text-ink"
          : block.level === 3
            ? "mb-1 mt-3 text-[13px] font-semibold text-ink"
            : "mb-1 mt-2.5 text-[13px] font-semibold text-ink";
      return (
        <p key={`h-${bi}`} role="heading" aria-level={block.level} className={size}>
          {inlineFormat(block.text)}
        </p>
      );
    }
    return (
      <div key={`p-${bi}`} className="whitespace-pre-wrap">
        {renderTextWithLinks(block.text)}
      </div>
    );
  });
}

function inlineFormat(content: string): ReactNode {
  const parts = formatSegment(content);
  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

// Convert each "Link: <url>" (or bare URL) into an inline clickable source chip.
function renderTextWithLinks(content: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const regex =
    /[ \t]*(?:source\s+link|link)\s*:?[ \t]*(https?:\/\/[^\s)\]]+)|(https?:\/\/[^\s)\]]+)/gi;
  let lastIndex = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(content)) !== null) {
    const rawUrl = m[1] ?? m[2] ?? "";
    const url = rawUrl.replace(/[.,;]+$/, "");
    const before = content.slice(lastIndex, m.index);
    if (before) {
      const parts = formatSegment(before);
      nodes.push(...parts.map((p, i) => <span key={`${key}-${i}`}>{p}</span>));
      key += 1;
    }
    const src = toSource(url);
    if (src) nodes.push(<InlineSourceChip key={key++} source={src} />);
    lastIndex = regex.lastIndex;
  }
  const rest = content.slice(lastIndex);
  if (rest) {
    const parts = formatSegment(rest);
    nodes.push(...parts.map((p, i) => <span key={`${key}-${i}`}>{p}</span>));
  }
  return nodes;
}

function InlineSourceChip({ source }: { source: Source }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      title={source.url}
      className="mx-0.5 inline-flex translate-y-0.5 items-center gap-1.5 rounded-full border border-black/10 bg-black/[0.04] px-2 py-0.5 text-[11px] font-medium text-ink/80 no-underline align-baseline transition hover:bg-black/[0.08] hover:text-ink"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`https://www.google.com/s2/favicons?sz=32&domain=${source.host}`}
        alt=""
        width={14}
        height={14}
        className="h-3.5 w-3.5 rounded-sm"
      />
      <span>{source.name}</span>
    </a>
  );
}

function toSource(url: string): Source | null {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    return { host, name: siteName(host), url };
  } catch {
    return null;
  }
}

const SITE_NAMES: Record<string, string> = {
  "99acres.com": "99acres",
  "magicbricks.com": "MagicBricks",
  "housing.com": "Housing",
  "squareyards.com": "Square Yards",
  "nobroker.in": "NoBroker",
  "commonfloor.com": "CommonFloor",
  "makaan.com": "Makaan",
  "proptiger.com": "PropTiger",
};

function siteName(host: string): string {
  if (SITE_NAMES[host]) return SITE_NAMES[host];
  const label = host.replace(/\.(com|in|org|net|co)(\.[a-z]+)?$/i, "").split(".").pop() ?? host;
  return label.charAt(0).toUpperCase() + label.slice(1);
}

// Light markdown for an inline text segment. Keeps ## out of body text;
// headings are handled as blocks. Renders **bold** as <strong>.
function formatSegment(content: string): ReactNode[] {
  const cleaned = content
    .replace(/__(.+?)__/g, "**$1**")
    .replace(/^[\t ]*[-*]\s+/gm, "• ");

  const nodes: ReactNode[] = [];
  const boldRe = /\*\*(.+?)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = boldRe.exec(cleaned)) !== null) {
    if (m.index > last) {
      nodes.push(stripSingleStars(cleaned.slice(last, m.index)));
    }
    nodes.push(
      <strong key={`b-${i++}`} className="font-semibold text-ink">
        {m[1]}
      </strong>,
    );
    last = boldRe.lastIndex;
  }
  if (last < cleaned.length) nodes.push(stripSingleStars(cleaned.slice(last)));
  return nodes;
}

function stripSingleStars(text: string): string {
  return text.replace(/\*(.+?)\*/g, "$1");
}

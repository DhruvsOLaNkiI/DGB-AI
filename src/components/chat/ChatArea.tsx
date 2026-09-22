"use client";

import { useEffect, useRef } from "react";
import { ANSWER_MODELS } from "@/lib/answer-model";
import type { ChatUsage } from "@/lib/chat-api";
import type { PandasEngine, RetrievalMode } from "@/lib/retrieval-mode";
import { WORD_LIMIT_OPTIONS } from "@/lib/word-limit";
import type { ChatMessage } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

type ChatAreaProps = {
  messages: ChatMessage[];
  isThinking: boolean;
  chatTitle: string;
  personalize: boolean;
  retrievalMode: RetrievalMode;
  pandasEngine: PandasEngine;
  wordLimit: number;
  answerModel: string;
  usage?: ChatUsage | null;
  onTogglePersonalize: () => void;
  onRetrievalModeChange: (mode: RetrievalMode) => void;
  onPandasEngineChange: (engine: PandasEngine) => void;
  onWordLimitChange: (limit: number) => void;
  onAnswerModelChange: (model: string) => void;
  onOpenSidebar: () => void;
  onNewChat: () => void;
  onQuoteReply: (text: string) => void;
};

export function ChatArea({
  messages,
  isThinking,
  chatTitle,
  personalize,
  retrievalMode,
  pandasEngine,
  wordLimit,
  answerModel,
  usage,
  onTogglePersonalize,
  onRetrievalModeChange,
  onPandasEngineChange,
  onWordLimitChange,
  onAnswerModelChange,
  onOpenSidebar,
  onNewChat,
  onQuoteReply,
}: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastUserImage = [...messages]
    .reverse()
    .find((m) => m.role === "user" && m.imageDataUrl)?.imageDataUrl;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-center gap-2 bg-transparent px-3 py-3 sm:gap-2.5 sm:px-5">
        <button
          type="button"
          className="shrink-0 rounded-lg p-2 text-white/90 hover:bg-white/10 hover:text-white md:hidden"
          onClick={onOpenSidebar}
          aria-label="Open sidebar"
        >
          <MenuIcon />
        </button>

        <button
          type="button"
          onClick={onNewChat}
          className="inline-flex min-w-0 max-w-[28%] shrink items-center gap-2 rounded-lg px-1 py-1.5 text-[13px] font-normal text-white drop-shadow-sm hover:opacity-90 sm:max-w-[22%]"
        >
          <HistoryIcon />
          <span className="truncate font-medium">
            {chatTitle === "New chat" ? "New AI chat" : chatTitle}
          </span>
          <ChevronIcon />
        </button>

        <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-2 overflow-x-auto sm:gap-2.5">
          <label className="hidden shrink-0 items-center sm:inline-flex">
            <span className="sr-only">Answer model</span>
            <select
              value={answerModel}
              onChange={(e) => onAnswerModelChange(e.target.value)}
              className="max-w-[10.5rem] rounded-full bg-black/25 px-2.5 py-1 text-[11px] font-medium text-white ring-1 ring-white/20 backdrop-blur-sm outline-none hover:bg-black/35 lg:max-w-[12rem] lg:text-[12px]"
              title="Model used for LLM answers"
              aria-label="Answer model"
            >
              {ANSWER_MODELS.map((m) => (
                <option key={m.id} value={m.id} className="text-ink">
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <div
            className="inline-flex shrink-0 items-center rounded-full bg-black/25 p-0.5 ring-1 ring-white/20 backdrop-blur-sm"
            role="group"
            aria-label="Retrieval mode"
          >
            <button
              type="button"
              onClick={() => onRetrievalModeChange("rag")}
              className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:px-2.5 sm:text-[11px] ${
                retrievalMode === "rag"
                  ? "bg-white text-ink shadow-sm"
                  : "text-white/80 hover:text-white"
              }`}
              title="Pinecone vector similarity search"
            >
              Vector RAG
            </button>
            <button
              type="button"
              onClick={() => onRetrievalModeChange("vectorless")}
              className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:px-2.5 sm:text-[11px] ${
                retrievalMode === "vectorless"
                  ? "bg-white text-ink shadow-sm"
                  : "text-white/80 hover:text-white"
              }`}
              title="PageIndex-style tree reasoning (no vectors)"
            >
              Vectorless
            </button>
            <button
              type="button"
              onClick={() => onRetrievalModeChange("pandas")}
              className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:px-2.5 sm:text-[11px] ${
                retrievalMode === "pandas"
                  ? "bg-white text-ink shadow-sm"
                  : "text-white/80 hover:text-white"
              }`}
              title="LangPanda: CSV inventory filters"
            >
              Pandas
            </button>
            <button
              type="button"
              onClick={() => onRetrievalModeChange("ask_dgb_sup")}
              className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:px-2.5 sm:text-[11px] ${
                retrievalMode === "ask_dgb_sup"
                  ? "bg-white text-ink shadow-sm"
                  : "text-white/80 hover:text-white"
              }`}
              title="ASK DGB-SUP: Gemini’s own answer only — no CSV, no web/DDGS"
            >
              ASK DGB-SUP
            </button>
          </div>

          {retrievalMode === "pandas" && (
            <div
              className="hidden shrink-0 items-center rounded-full bg-black/25 p-0.5 ring-1 ring-white/20 backdrop-blur-sm sm:inline-flex"
              role="group"
              aria-label="Pandas engine"
            >
              <button
                type="button"
                onClick={() => onPandasEngineChange("pandas_only")}
                className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:text-[11px] ${
                  pandasEngine === "pandas_only"
                    ? "bg-white text-ink shadow-sm"
                    : "text-white/80 hover:text-white"
                }`}
                title="Exact CSV filters only — fast, no LLM"
              >
                Only Pandas
              </button>
              <button
                type="button"
                onClick={() => onPandasEngineChange("pandas_llm")}
                className={`whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition sm:text-[11px] ${
                  pandasEngine === "pandas_llm"
                    ? "bg-white text-ink shadow-sm"
                    : "text-white/80 hover:text-white"
                }`}
                title="CSV first, then UI LLM + web fallback when needed"
              >
                Pandas + LLM
              </button>
            </div>
          )}

          {retrievalMode === "ask_dgb_sup" && (
            <label
              className="hidden shrink-0 items-center gap-1 rounded-full bg-black/25 px-2 py-0.5 text-[10px] font-medium text-white ring-1 ring-white/20 backdrop-blur-sm sm:inline-flex sm:text-[11px]"
              title="Max words for ASK DGB-SUP answers"
            >
              <span className="text-white/75">Words</span>
              <select
                value={wordLimit}
                onChange={(e) => onWordLimitChange(Number(e.target.value))}
                className="max-w-[6.5rem] cursor-pointer rounded-full border-0 bg-transparent py-0.5 text-[10px] font-medium text-white outline-none sm:text-[11px]"
                aria-label="ASK DGB-SUP word limit"
              >
                {WORD_LIMIT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value} className="text-ink">
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="hidden shrink-0 items-center gap-2 text-[12px] text-white drop-shadow-sm xl:inline-flex">
            Personalize
            <button
              type="button"
              role="switch"
              aria-checked={personalize}
              onClick={onTogglePersonalize}
              className={`relative h-5 w-9 rounded-full transition ${
                personalize ? "bg-white" : "bg-white/35"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full transition ${
                  personalize ? "translate-x-4 bg-accent" : "bg-white"
                }`}
              />
            </button>
          </label>

          <div className="flex shrink-0 items-center gap-0.5 text-white drop-shadow-sm">
            <IconButton label="Edit">
              <PencilIcon />
            </IconButton>
            <IconButton label="Focus">
              <FocusIcon />
            </IconButton>
            <IconButton label="Panel" onClick={onOpenSidebar}>
              <PanelIcon />
            </IconButton>
          </div>
        </div>
      </header>

      <div className="flex flex-col gap-1 px-3 pb-1 sm:hidden">
        <select
          value={answerModel}
          onChange={(e) => onAnswerModelChange(e.target.value)}
          className="w-full rounded-xl bg-black/25 px-3 py-2 text-[12px] font-medium text-white ring-1 ring-white/20 outline-none"
          aria-label="Answer model"
        >
          {ANSWER_MODELS.map((m) => (
            <option key={m.id} value={m.id} className="text-ink">
              {m.label}
            </option>
          ))}
        </select>
        {retrievalMode === "pandas" && (
          <div
            className="inline-flex w-full items-center rounded-xl bg-black/25 p-0.5 ring-1 ring-white/20"
            role="group"
            aria-label="Pandas engine"
          >
            <button
              type="button"
              onClick={() => onPandasEngineChange("pandas_only")}
              className={`flex-1 rounded-xl px-2 py-1.5 text-[11px] font-medium ${
                pandasEngine === "pandas_only"
                  ? "bg-white text-ink"
                  : "text-white/80"
              }`}
            >
              Only Pandas
            </button>
            <button
              type="button"
              onClick={() => onPandasEngineChange("pandas_llm")}
              className={`flex-1 rounded-xl px-2 py-1.5 text-[11px] font-medium ${
                pandasEngine === "pandas_llm"
                  ? "bg-white text-ink"
                  : "text-white/80"
              }`}
            >
              Pandas + LLM
            </button>
          </div>
        )}
        {retrievalMode === "ask_dgb_sup" && (
          <select
            value={wordLimit}
            onChange={(e) => onWordLimitChange(Number(e.target.value))}
            className="w-full rounded-xl bg-black/25 px-3 py-2 text-[12px] font-medium text-white ring-1 ring-white/20 outline-none"
            aria-label="ASK DGB-SUP word limit"
          >
            {WORD_LIMIT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} className="text-ink">
                Word limit · {opt.label}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="relative flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        {lastUserImage && (
          <div className="pointer-events-none absolute top-6 right-6 z-10 hidden w-44 flex-col items-end gap-2 sm:flex">
            <div className="glass-strong w-full rotate-2 rounded-xl p-2 shadow-xl">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={lastUserImage}
                alt=""
                className="h-20 w-full rounded-lg object-cover"
              />
              <p className="mt-1.5 truncate px-1 text-[11px] text-ink/60">
                Listing photo
              </p>
            </div>
            <span className="rounded-full bg-white px-3 py-1.5 text-[11px] text-ink shadow-sm">
              Property context
            </span>
          </div>
        )}

        <div className="mx-auto flex w-full max-w-3xl flex-col gap-8">
          {messages.length === 0 && !isThinking ? (
            <EmptyState />
          ) : (
            messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                onQuoteReply={m.role === "assistant" ? onQuoteReply : undefined}
              />
            ))
          )}

          {isThinking && (
            <div className="flex items-center gap-2 text-[12px] font-normal text-ink/70">
              <span className="thinking-ring" />
              Getting a detailed report…
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col py-10">
      <p className="max-w-xl text-[13px] font-normal leading-6 text-ink">
        Hello! I&apos;m DBG-AI, your real estate assistant. Ask about buying,
        selling, renting, neighborhoods, or listing photos.
      </p>
      <ul className="mt-6 grid max-w-xl gap-2 text-[12px] text-ink sm:grid-cols-2">
        {[
          "What should I ask before making an offer?",
          "Compare buying vs renting for 5 years",
          "How do I read days on market?",
          "Upload a photo of a living room",
        ].map((hint) => (
          <li
            key={hint}
            className="rounded-full bg-white px-4 py-3 shadow-sm ring-1 ring-black/5"
          >
            {hint}
          </li>
        ))}
      </ul>
    </div>
  );
}

function IconButton({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className="rounded-lg p-2 text-white/90 hover:bg-white/10 hover:text-white"
    >
      {children}
    </button>
  );
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 7h16M4 12h16M4 17h16"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 8v4l3 2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M6 9l6 6 6-6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 20h4L19 9l-4-4L4 16v4z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function FocusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PanelIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect
        x="3"
        y="4"
        width="18"
        height="16"
        rx="2"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path d="M9 4v16" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

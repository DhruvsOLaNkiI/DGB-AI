"use client";

import { useEffect, useRef, useState } from "react";

type ComposerProps = {
  disabled?: boolean;
  draft?: string;
  onDraftConsumed?: () => void;
  onSend: (payload: { text: string; imageDataUrl?: string }) => void;
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
};

type SpeechRecognitionEventLike = {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
};

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  }
}

export function Composer({
  disabled,
  draft,
  onDraftConsumed,
  onSend,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [imageDataUrl, setImageDataUrl] = useState<string | undefined>();
  const [listening, setListening] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (!draft) return;
    setText(draft);
    onDraftConsumed?.();
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
      el.focus();
    });
  }, [draft, onDraftConsumed]);

  function resizeTextarea() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function handleFile(file: File | undefined) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") setImageDataUrl(reader.result);
    };
    reader.readAsDataURL(file);
  }

  function toggleVoice() {
    setVoiceError(null);
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Ctor) {
      setVoiceError("Voice input needs Chrome or Edge on this device.");
      return;
    }

    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }

    const recognition = new Ctor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0]?.transcript ?? "")
        .join("")
        .trim();
      if (transcript) {
        setText((prev) => (prev ? `${prev.trim()} ${transcript}` : transcript));
        requestAnimationFrame(resizeTextarea);
      }
    };
    recognition.onerror = (event) => {
      setListening(false);
      if (event.error !== "aborted") {
        setVoiceError("Couldn’t hear that — try again or type your question.");
      }
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  function submit() {
    const trimmed = text.trim();
    if (disabled) return;
    if (!trimmed && !imageDataUrl) return;
    onSend({ text: trimmed, imageDataUrl });
    setText("");
    setImageDataUrl(undefined);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  return (
    <div className="px-3 py-3 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        {imageDataUrl && (
          <div className="glass-strong mb-2 inline-flex items-start gap-2 rounded-xl px-2 py-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageDataUrl}
              alt="Preview"
              className="h-16 w-16 rounded-lg object-cover"
            />
            <button
              type="button"
              className="rounded p-1 text-ink/50 hover:bg-black/5 hover:text-ink"
              onClick={() => setImageDataUrl(undefined)}
              aria-label="Remove photo"
            >
              ×
            </button>
          </div>
        )}

        <div className="flex items-end gap-2 rounded-[1.75rem] bg-white p-2.5 shadow-[0_8px_30px_rgba(0,0,0,0.12),0_2px_8px_rgba(249,115,22,0.12)] ring-1 ring-black/5">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              handleFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            disabled={disabled}
            onClick={() => fileRef.current?.click()}
            className="shrink-0 rounded-xl p-2.5 text-ink/50 transition hover:bg-black/5 hover:text-ink disabled:opacity-40"
            aria-label="Upload photo"
            title="Upload photo"
          >
            <PhotoIcon />
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            disabled={disabled}
            placeholder="Ask Me Anything…"
            className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-1 py-2 text-[13px] font-normal text-ink outline-none placeholder:text-ink/40"
            onChange={(e) => {
              setText(e.target.value);
              resizeTextarea();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />

          <button
            type="button"
            disabled={disabled}
            onClick={toggleVoice}
            className={`shrink-0 rounded-xl p-2.5 transition disabled:opacity-40 ${
              listening
                ? "bg-accent-soft text-accent"
                : "text-ink/50 hover:bg-black/5 hover:text-ink"
            }`}
            aria-label={listening ? "Stop voice" : "Voice input"}
            title={listening ? "Stop voice" : "Voice input"}
          >
            <MicIcon active={listening} />
          </button>

          <button
            type="button"
            disabled={disabled || (!text.trim() && !imageDataUrl)}
            onClick={submit}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-accent text-white transition hover:bg-teal-bright disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </div>

        {voiceError && (
          <p className="mt-2 text-xs text-rose-600">{voiceError}</p>
        )}
        {listening && (
          <p className="mt-2 text-xs font-medium text-accent">Listening… speak your question</p>
        )}
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M5 12h14M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PhotoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="9" cy="11" r="2" stroke="currentColor" strokeWidth="1.8" />
      <path d="M13 15l3-3 4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MicIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect
        x="9"
        y="3"
        width="6"
        height="11"
        rx="3"
        stroke="currentColor"
        strokeWidth="1.8"
        fill={active ? "currentColor" : "none"}
      />
      <path
        d="M6 11a6 6 0 0012 0M12 17v4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

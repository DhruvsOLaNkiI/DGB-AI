"use client";

import { useCallback, useEffect, useState } from "react";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { Composer } from "@/components/chat/Composer";
import {
  DOC_KEY,
  VectorlessDocBar,
} from "@/components/chat/VectorlessDocBar";
import {
  ANSWER_MODEL_KEY,
  ANSWER_MODELS,
  parseAnswerModel,
} from "@/lib/answer-model";
import {
  createEmptyChat,
  createId,
  loadChatStore,
  saveChatStore,
  titleFromPrompt,
} from "@/lib/chat-store";
import { sendChatRequest, type ChatUsage } from "@/lib/chat-api";
import {
  PANDAS_ENGINE_KEY,
  RETRIEVAL_MODE_KEY,
  parsePandasEngine,
  parseRetrievalMode,
  type PandasEngine,
  type RetrievalMode,
} from "@/lib/retrieval-mode";
import {
  WORD_LIMIT_KEY,
  defaultWordLimit,
  parseWordLimit,
} from "@/lib/word-limit";
import type { Chat, ChatMessage } from "@/lib/types";

const SESSION_KEY = "dbg-ai-session-id";
const PERSONALIZE_KEY = "dbg-ai-personalize";

function getOrCreateSessionId(): string {
  const existing = window.localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const id = createId("session");
  window.localStorage.setItem(SESSION_KEY, id);
  return id;
}

export function ChatShell() {
  const [ready, setReady] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [usage, setUsage] = useState<ChatUsage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [personalize, setPersonalize] = useState(true);
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("rag");
  const [pandasEngine, setPandasEngine] =
    useState<PandasEngine>("pandas_only");
  const [wordLimit, setWordLimit] = useState(defaultWordLimit);
  const [answerModel, setAnswerModel] = useState(
    () => ANSWER_MODELS[0]?.id ?? "gemini-3.6-flash",
  );
  const [vectorlessDocId, setVectorlessDocId] = useState<string | null>(null);
  const [composerDraft, setComposerDraft] = useState<string | undefined>();

  useEffect(() => {
    const stored = loadChatStore();
    if (stored.chats.length === 0) {
      const chat = createEmptyChat();
      setChats([chat]);
      setActiveChatId(chat.id);
    } else {
      setChats(stored.chats);
      setActiveChatId(stored.activeChatId ?? stored.chats[0]!.id);
    }
    setSessionId(getOrCreateSessionId());
    const savedPersonalize = window.localStorage.getItem(PERSONALIZE_KEY);
    if (savedPersonalize != null) setPersonalize(savedPersonalize === "1");
    setRetrievalMode(
      parseRetrievalMode(window.localStorage.getItem(RETRIEVAL_MODE_KEY)),
    );
    setPandasEngine(
      parsePandasEngine(window.localStorage.getItem(PANDAS_ENGINE_KEY)),
    );
    setWordLimit(parseWordLimit(window.localStorage.getItem(WORD_LIMIT_KEY)));
    setAnswerModel(
      parseAnswerModel(window.localStorage.getItem(ANSWER_MODEL_KEY)),
    );
    const savedDoc = window.localStorage.getItem(DOC_KEY);
    if (savedDoc) setVectorlessDocId(savedDoc);
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    saveChatStore({ chats, activeChatId });
  }, [chats, activeChatId, ready]);

  const activeChat = chats.find((c) => c.id === activeChatId) ?? null;

  const updateChat = useCallback((chatId: string, updater: (chat: Chat) => Chat) => {
    setChats((prev) => prev.map((c) => (c.id === chatId ? updater(c) : c)));
  }, []);

  function handleNewChat() {
    const chat = createEmptyChat();
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
    setError(null);
  }

  function handleDeleteChat(id: string) {
    setChats((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = createEmptyChat();
        setActiveChatId(fresh.id);
        return [fresh];
      }
      if (activeChatId === id) {
        setActiveChatId(next[0]!.id);
      }
      return next;
    });
  }

  function handleTogglePersonalize() {
    setPersonalize((prev) => {
      const next = !prev;
      window.localStorage.setItem(PERSONALIZE_KEY, next ? "1" : "0");
      return next;
    });
  }

  function handleRetrievalModeChange(mode: RetrievalMode) {
    setRetrievalMode(mode);
    window.localStorage.setItem(RETRIEVAL_MODE_KEY, mode);
  }

  function handlePandasEngineChange(engine: PandasEngine) {
    setPandasEngine(engine);
    window.localStorage.setItem(PANDAS_ENGINE_KEY, engine);
  }

  function handleWordLimitChange(limit: number) {
    const next = parseWordLimit(limit);
    setWordLimit(next);
    window.localStorage.setItem(WORD_LIMIT_KEY, String(next));
  }

  function handleAnswerModelChange(model: string) {
    const next = parseAnswerModel(model);
    setAnswerModel(next);
    window.localStorage.setItem(ANSWER_MODEL_KEY, next);
  }

  function handleQuoteReply(text: string) {
    const clipped = text.length > 280 ? `${text.slice(0, 280)}…` : text;
    setComposerDraft(`Regarding: “${clipped}”\n\n`);
  }

  async function handleSend(payload: { text: string; imageDataUrl?: string }) {
    let chatId = activeChatId;
    let workingMessages: ChatMessage[] = activeChat?.messages ?? [];

    if (!chatId) {
      const chat = createEmptyChat();
      setChats((prev) => [chat, ...prev]);
      chatId = chat.id;
      setActiveChatId(chat.id);
      workingMessages = [];
    }

    const personalizedPrefix = personalize
      ? ""
      : "[Respond in a neutral, non-personalized tone.]\n\n";

    const userMessage: ChatMessage = {
      id: createId("msg"),
      role: "user",
      content: payload.text,
      imageDataUrl: payload.imageDataUrl,
      createdAt: Date.now(),
    };

    const apiMessages: ChatMessage[] = [
      ...workingMessages,
      {
        ...userMessage,
        content: personalizedPrefix + payload.text,
      },
    ];

    updateChat(chatId, (chat) => {
      const isFirst = chat.messages.length === 0;
      return {
        ...chat,
        title: isFirst ? titleFromPrompt(payload.text) : chat.title,
        messages: [...chat.messages, userMessage],
        updatedAt: Date.now(),
      };
    });

    setIsThinking(true);
    setError(null);

    try {
      const result = await sendChatRequest({
        messages: apiMessages,
        sessionId,
        mode: retrievalMode,
        docId: retrievalMode === "vectorless" ? vectorlessDocId : null,
        model: answerModel,
        pandasEngine:
          retrievalMode === "pandas" ? pandasEngine : undefined,
        wordLimit: retrievalMode === "ask_dgb_sup" ? wordLimit : undefined,
      });

      if (result.usage) setUsage(result.usage);

      if (!result.ok) {
        setError(result.error);
        const assistantMessage: ChatMessage = {
          id: createId("msg"),
          role: "assistant",
          content: `I couldn’t complete that request.\n\n${result.error}`,
          createdAt: Date.now(),
        };
        updateChat(chatId, (chat) => ({
          ...chat,
          messages: [...chat.messages, assistantMessage],
          updatedAt: Date.now(),
        }));
        return;
      }

      const assistantMessage: ChatMessage = {
        id: createId("msg"),
        role: "assistant",
        content: result.reply,
        createdAt: Date.now(),
        deliverySource: result.source,
        pandasEngine:
          result.pandasEngine ??
          (retrievalMode === "pandas" ? pandasEngine : undefined),
      };
      updateChat(chatId, (chat) => ({
        ...chat,
        messages: [...chat.messages, assistantMessage],
        updatedAt: Date.now(),
      }));
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Network error talking to DBG-AI.";
      setError(message);
    } finally {
      setIsThinking(false);
    }
  }

  if (!ready) {
    return (
      <div className="flex h-dvh items-center justify-center bg-[#0a0a0a] text-white/60">
        Loading DBG-AI…
      </div>
    );
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-[#0a0a0a] text-ink">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        open={sidebarOpen}
        onCloseMobile={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        onSelectChat={setActiveChatId}
        onRenameChat={(id, title) =>
          updateChat(id, (c) => ({ ...c, title, updatedAt: Date.now() }))
        }
        onDeleteChat={handleDeleteChat}
      />

      <div className="flex min-w-0 flex-1 p-2 md:p-3 md:pl-0">
        <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-[1.75rem] shadow-[0_0_0_1px_rgba(255,255,255,0.06),0_20px_50px_rgba(0,0,0,0.45)]">
          <div className="pointer-events-none absolute inset-0 bg-atmosphere" aria-hidden />
          <div className="relative flex min-h-0 flex-1 flex-col">
            <ChatArea
              messages={activeChat?.messages ?? []}
              isThinking={isThinking}
              chatTitle={activeChat?.title ?? "New chat"}
              personalize={personalize}
              retrievalMode={retrievalMode}
              pandasEngine={pandasEngine}
              wordLimit={wordLimit}
              answerModel={answerModel}
              usage={usage}
              onTogglePersonalize={handleTogglePersonalize}
              onRetrievalModeChange={handleRetrievalModeChange}
              onPandasEngineChange={handlePandasEngineChange}
              onWordLimitChange={handleWordLimitChange}
              onAnswerModelChange={handleAnswerModelChange}
              onOpenSidebar={() => setSidebarOpen(true)}
              onNewChat={handleNewChat}
              onQuoteReply={handleQuoteReply}
            />
            {error && (
              <div className="mx-3 mb-1 rounded-xl bg-rose-500/15 px-4 py-2 text-xs text-rose-800 ring-1 ring-rose-500/20 sm:mx-6">
                {error}
              </div>
            )}
            <VectorlessDocBar
              mode={retrievalMode}
              docId={vectorlessDocId}
              onDocIdChange={setVectorlessDocId}
            />
            <Composer
              disabled={isThinking}
              draft={composerDraft}
              onDraftConsumed={() => setComposerDraft(undefined)}
              onSend={handleSend}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

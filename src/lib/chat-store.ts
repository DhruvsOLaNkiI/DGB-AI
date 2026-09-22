import type { Chat, ChatStoreSnapshot } from "./types";

const STORAGE_KEY = "dbg-ai-chats-v1";

export function createId(prefix = "id"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
}

export function loadChatStore(): ChatStoreSnapshot {
  if (typeof window === "undefined") {
    return { chats: [], activeChatId: null };
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { chats: [], activeChatId: null };
    const parsed = JSON.parse(raw) as ChatStoreSnapshot;
    if (!parsed || !Array.isArray(parsed.chats)) {
      return { chats: [], activeChatId: null };
    }
    return {
      chats: parsed.chats,
      activeChatId: parsed.activeChatId ?? parsed.chats[0]?.id ?? null,
    };
  } catch {
    return { chats: [], activeChatId: null };
  }
}

export function saveChatStore(snapshot: ChatStoreSnapshot): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
}

export function createEmptyChat(): Chat {
  const now = Date.now();
  return {
    id: createId("chat"),
    title: "New chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function titleFromPrompt(text: string): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return "Photo question";
  return cleaned.length > 42 ? `${cleaned.slice(0, 42)}…` : cleaned;
}

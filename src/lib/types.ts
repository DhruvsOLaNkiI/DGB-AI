export type MessageRole = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  imageDataUrl?: string;
  createdAt: number;
  /** How the assistant answer was produced (e.g. pandas-llm, web-fallback). */
  deliverySource?: string;
  pandasEngine?: "pandas_only" | "pandas_llm";
};

export type Chat = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
};

export type ChatStoreSnapshot = {
  chats: Chat[];
  activeChatId: string | null;
};

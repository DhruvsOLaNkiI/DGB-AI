"use client";

import type { Chat } from "@/lib/types";

type SidebarProps = {
  chats: Chat[];
  activeChatId: string | null;
  open: boolean;
  onCloseMobile: () => void;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onRenameChat: (id: string, title: string) => void;
  onDeleteChat: (id: string) => void;
};

export function Sidebar({
  chats,
  activeChatId,
  open,
  onCloseMobile,
  onNewChat,
  onSelectChat,
  onRenameChat,
  onDeleteChat,
}: SidebarProps) {
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-black/70 transition-opacity md:hidden ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={onCloseMobile}
        aria-hidden={!open}
      />

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[260px] flex-col bg-[#0a0a0a] text-white transition-transform md:static md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between gap-3 px-5 pb-2 pt-6">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-sm font-bold text-white">
              D
            </span>
            <div>
              <p className="text-[15px] font-medium tracking-tight text-white">
                DBG-AI
              </p>
              <p className="text-[11px] text-white/45">Real estate assistant</p>
            </div>
          </div>
          <button
            type="button"
            className="rounded-md p-2 text-white/50 hover:bg-white/8 hover:text-white md:hidden"
            onClick={onCloseMobile}
            aria-label="Close sidebar"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="px-4 pb-4 pt-3">
          <button
            type="button"
            onClick={() => {
              onNewChat();
              onCloseMobile();
            }}
            className="flex w-full items-center gap-2 rounded-xl bg-[#1c1c1c] px-3.5 py-2.5 text-[13px] font-medium text-white ring-1 ring-white/8 transition hover:bg-[#242424]"
          >
            <PlusIcon />
            New chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-4">
          <p className="px-2 pb-2 text-[11px] font-medium uppercase tracking-wider text-white/35">
            Recent
          </p>
          {sorted.length === 0 ? (
            <p className="px-2 text-sm text-white/35">No chats yet</p>
          ) : (
            <ul className="space-y-0.5">
              {sorted.map((chat) => {
                const active = chat.id === activeChatId;
                return (
                  <li key={chat.id} className="group relative">
                    <button
                      type="button"
                      onClick={() => {
                        onSelectChat(chat.id);
                        onCloseMobile();
                      }}
                      className={`w-full rounded-xl px-3 py-2.5 text-left text-[13px] transition ${
                        active
                          ? "bg-white/10 text-white"
                          : "text-white/55 hover:bg-white/5 hover:text-white/85"
                      }`}
                    >
                      <span className="line-clamp-1 pr-14 font-medium">
                        {chat.title}
                      </span>
                      <span className="mt-0.5 block text-[11px] text-white/30">
                        {formatRelative(chat.updatedAt)}
                      </span>
                    </button>
                    <div className="absolute right-1 top-1.5 flex gap-0.5 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                      <button
                        type="button"
                        className="rounded p-1.5 text-white/40 hover:bg-white/10 hover:text-white"
                        title="Rename"
                        aria-label={`Rename ${chat.title}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          const next = window.prompt("Rename chat", chat.title);
                          if (next && next.trim()) onRenameChat(chat.id, next.trim());
                        }}
                      >
                        <PencilIcon />
                      </button>
                      <button
                        type="button"
                        className="rounded p-1.5 text-white/40 hover:bg-rose-500/20 hover:text-rose-300"
                        title="Delete"
                        aria-label={`Delete ${chat.title}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm("Delete this chat?")) {
                            onDeleteChat(chat.id);
                          }
                        }}
                      >
                        <TrashIcon />
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="mt-auto px-4 pb-5">
          <div className="rounded-2xl bg-[#141414] p-4 ring-1 ring-white/6">
            <p className="text-sm text-white/70">DBG-AI · retrieval modes</p>
            <p className="mt-1 text-xs text-white/35">
              Switch Vector RAG, Vectorless, Pandas, or ASK DGB-SUP in the header.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

function formatRelative(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 20h4L19 9l-4-4L4 16v4z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M5 7h14M10 11v6M14 11v6M8 7l1-2h6l1 2M7 7l1 13h8l1-13"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

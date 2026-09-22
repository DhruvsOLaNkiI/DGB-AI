type UsageBucket = {
  dayKey: string;
  tokensUsed: number;
};

const globalForUsage = globalThis as typeof globalThis & {
  __dbgAiUsage?: Map<string, UsageBucket>;
};

function store() {
  if (!globalForUsage.__dbgAiUsage) {
    globalForUsage.__dbgAiUsage = new Map();
  }
  return globalForUsage.__dbgAiUsage;
}

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

export function getSessionUsage(sessionId: string): number {
  const entry = store().get(sessionId);
  if (!entry || entry.dayKey !== todayKey()) return 0;
  return entry.tokensUsed;
}

export function addSessionUsage(sessionId: string, tokens: number): number {
  const dayKey = todayKey();
  const current = store().get(sessionId);
  const base =
    current && current.dayKey === dayKey ? current.tokensUsed : 0;
  const next = base + Math.max(0, tokens);
  store().set(sessionId, { dayKey, tokensUsed: next });
  return next;
}

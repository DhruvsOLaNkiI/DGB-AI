/** Word-limit options for ASK DGB-SUP answers. */

export type WordLimitOption = {
  /** 0 = no limit */
  value: number;
  label: string;
};

export const WORD_LIMIT_KEY = "dbg-ai-ask-word-limit";

export const WORD_LIMIT_OPTIONS: WordLimitOption[] = [
  { value: 100, label: "100 words" },
  { value: 200, label: "200 words" },
  { value: 300, label: "300 words" },
  { value: 500, label: "500 words" },
  { value: 800, label: "800 words" },
  { value: 0, label: "No limit" },
];

const ALLOWED = new Set(WORD_LIMIT_OPTIONS.map((o) => o.value));

export function defaultWordLimit(): number {
  return 300;
}

export function parseWordLimit(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value) && ALLOWED.has(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const n = Number(value.trim());
    if (Number.isFinite(n) && ALLOWED.has(n)) return n;
  }
  return defaultWordLimit();
}

export function wordLimitLabel(value: number): string {
  return WORD_LIMIT_OPTIONS.find((o) => o.value === value)?.label ?? `${value} words`;
}

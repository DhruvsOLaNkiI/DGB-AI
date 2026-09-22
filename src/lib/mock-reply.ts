const REPLIES = [
  "I can help with that. Looking at typical market patterns, homes in that range often move faster when priced near recent comps. Share a neighborhood or budget and I’ll narrow it down.",
  "Good question. For buyers, I’d check days-on-market, price cuts, and comparable sales within about half a mile. Want me to walk through what those numbers usually mean?",
  "From a listing perspective: strong photos, clear floor plans, and transparent HOA/fees tend to get more serious inquiries. Tell me the property type and I can suggest a simple checklist.",
  "If you’re comparing rent vs buy, the break-even often depends on how long you’ll stay, down payment, and local appreciation. Share your city and timeline and I’ll outline a simple framework.",
  "Happy to look at a photo. I can comment on layout cues, condition signals, and questions to ask the agent — this is a demo reply until live AI is connected in Phase 2.",
];

export async function getMockReply(input: {
  text: string;
  hasImage: boolean;
}): Promise<string> {
  await new Promise((r) => setTimeout(r, 700 + Math.random() * 600));

  if (input.hasImage && !input.text.trim()) {
    return "Thanks for the photo. In this frontend demo I can’t analyze images yet — once Gemini is wired, I’ll describe layout, condition cues, and follow-up questions for the listing.";
  }

  if (input.hasImage) {
    return `Got your photo and note (“${short(input.text)}”). Demo mode: I’ll treat this as a property visual. Next phase will use Gemini vision for a real read. Meanwhile — what’s your budget and preferred area?`;
  }

  const pick = REPLIES[Math.floor(Math.random() * REPLIES.length)]!;
  return `${pick}\n\n(You asked: “${short(input.text)}” — mock reply; live Gemini answers come in Phase 2.)`;
}

function short(text: string): string {
  const t = text.trim();
  return t.length > 80 ? `${t.slice(0, 80)}…` : t;
}

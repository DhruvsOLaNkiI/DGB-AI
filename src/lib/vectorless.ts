import {
  type ListingFilters,
  type RetrievedListing,
  getPriceSort,
  listingMatchesAllFilters,
  parseListingFilters,
} from "@/lib/rag";
import {
  extractSector,
  loadListingsFromCsv,
  type ListingRecord,
} from "@/lib/listings";
import { isOllamaAnswerModel } from "@/lib/answer-model";
import {
  createGeminiClient,
  getGeminiConfig,
} from "@/lib/gemini";
import { ollamaGenerate } from "@/lib/ollama";

export type TreeNodeSummary = {
  sector: string;
  bedroom: number | null;
  count: number;
  minPriceLakh: number | null;
  maxPriceLakh: number | null;
};

type ReasonedFilters = ListingFilters & {
  sectors?: string[];
};

let treeCache: TreeNodeSummary[] | null = null;

function listingSector(listing: ListingRecord): string {
  return listing.sector || extractSector(listing.address) || "other";
}

/** Hierarchical outline of the corpus — PageIndex-style tree index, no vectors. */
export function buildListingsTreeOutline(): TreeNodeSummary[] {
  if (treeCache) return treeCache;

  const buckets = new Map<string, TreeNodeSummary>();

  for (const listing of loadListingsFromCsv()) {
    const sector = listingSector(listing);
    const bedroom = listing.bedroom;
    const key = `${sector}::${bedroom ?? "na"}`;
    const existing = buckets.get(key);
    const price = listing.priceInLakh;

    if (!existing) {
      buckets.set(key, {
        sector,
        bedroom,
        count: 1,
        minPriceLakh: price,
        maxPriceLakh: price,
      });
      continue;
    }

    existing.count += 1;
    if (price != null) {
      existing.minPriceLakh =
        existing.minPriceLakh == null
          ? price
          : Math.min(existing.minPriceLakh, price);
      existing.maxPriceLakh =
        existing.maxPriceLakh == null
          ? price
          : Math.max(existing.maxPriceLakh, price);
    }
  }

  treeCache = [...buckets.values()].sort((a, b) => {
    const sectorCmp = a.sector.localeCompare(b.sector, undefined, {
      numeric: true,
    });
    if (sectorCmp !== 0) return sectorCmp;
    return (a.bedroom ?? 999) - (b.bedroom ?? 999);
  });

  return treeCache;
}

function formatTreeOutline(nodes: TreeNodeSummary[]): string {
  const bySector = new Map<string, TreeNodeSummary[]>();
  for (const node of nodes) {
    const list = bySector.get(node.sector) ?? [];
    list.push(node);
    bySector.set(node.sector, list);
  }

  const lines: string[] = [];
  for (const [sector, children] of bySector) {
    const total = children.reduce((sum, n) => sum + n.count, 0);
    lines.push(`# ${sector} (${total} listings)`);
    for (const child of children) {
      const bhk =
        child.bedroom != null ? `${child.bedroom} BHK` : "Other / unspecified";
      const price =
        child.minPriceLakh != null && child.maxPriceLakh != null
          ? ` · ${child.minPriceLakh.toFixed(1)}–${child.maxPriceLakh.toFixed(1)} lakh`
          : "";
      lines.push(`  - ${bhk}: ${child.count} listings${price}`);
    }
  }
  return lines.join("\n");
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const raw = (fenced?.[1] ?? text).trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(raw.slice(start, end + 1)) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** LLM reasons over the tree outline (vectorless) to pick relevant branches. */
export async function reasonTreeFilters(
  question: string,
  modelOverride?: string,
): Promise<ReasonedFilters> {
  const fallback = parseListingFilters(question);
  const outline = formatTreeOutline(buildListingsTreeOutline());
  const prompt = `You navigate a hierarchical tree index of Noida real-estate listings (vectorless retrieval).
Given the user question and the TREE OUTLINE, choose which branches to open.

Return ONLY valid JSON with this shape:
{
  "sectors": ["sector 62", "sector 75"],
  "bedroom": 2,
  "maxPriceLakh": 80
}

Rules:
- sectors: array of sector labels exactly as in the outline (lowercase like "sector 62"). Empty array if none specified.
- bedroom: integer BHK if asked, else null
- maxPriceLakh: number if a budget/under-price is asked, else null
- Prefer precision over guessing. Do not invent sectors that are not in the outline.

QUESTION:
${question}

TREE OUTLINE:
${outline.slice(0, 120_000)}`;

  try {
    let text = "";
    if (modelOverride && isOllamaAnswerModel(modelOverride)) {
      text = await ollamaGenerate({
        model: modelOverride,
        prompt,
        temperature: 0,
      });
    } else {
      const config = getGeminiConfig(modelOverride);
      if (!config.apiKey) return fallback;
      const genAI = createGeminiClient(config.apiKey);
      const model = genAI.getGenerativeModel({
        model: config.model,
        generationConfig: {
          temperature: 0,
          maxOutputTokens: 256,
        },
      });
      const result = await model.generateContent(prompt);
      text = result.response.text()?.trim() || "";
    }

    const json = parseJsonObject(text);
    if (!json) return fallback;

    const reasoned: ReasonedFilters = { ...fallback };

    if (Array.isArray(json.sectors)) {
      const sectors = json.sectors
        .map((s) => String(s).trim().toLowerCase())
        .filter((s) => /^sector\s+\d+[a-z]?$/.test(s));
      if (sectors.length > 0) {
        reasoned.sectors = [...new Set(sectors)];
        reasoned.sectorQuery = sectors[0];
      }
    }

    if (json.bedroom != null && Number.isFinite(Number(json.bedroom))) {
      reasoned.bedroom = Number(json.bedroom);
    }

    if (
      json.maxPriceLakh != null &&
      Number.isFinite(Number(json.maxPriceLakh))
    ) {
      reasoned.maxPriceLakh = Number(json.maxPriceLakh);
    }

    return reasoned;
  } catch (error) {
    console.warn(
      "[vectorless] tree reasoning failed, using regex filters:",
      error instanceof Error ? error.message : error,
    );
    return fallback;
  }
}

function matchesReasonedFilters(
  listing: ListingRecord,
  filters: ReasonedFilters,
): boolean {
  // Sector multi-select is OR within sectors; all other conditions are AND.
  if (filters.sectors && filters.sectors.length > 0) {
    const sector = listingSector(listing);
    const sectorOk = filters.sectors.some(
      (wanted) =>
        sector === wanted || listing.address.toLowerCase().includes(wanted),
    );
    if (!sectorOk) return false;
    const { sectors: _sectors, sectorQuery: _sq, ...rest } = filters;
    return listingMatchesAllFilters(listing, rest);
  }

  return listingMatchesAllFilters(listing, filters);
}

function toRetrieved(listing: ListingRecord, score: number): RetrievedListing {
  return {
    id: listing.id,
    score,
    rawScore: score,
    text: listing.text,
    address: listing.address || undefined,
    sector: listingSector(listing) || undefined,
    bedroom: listing.bedroom ?? undefined,
    priceInLakh: listing.priceInLakh ?? undefined,
    type2: listing.type2 || undefined,
    status: listing.status || undefined,
    furnishing: listing.furnishing || undefined,
  };
}

/**
 * Vectorless retrieval: reason over a hierarchical tree of listings,
 * then open matching leaf nodes — no embeddings / Pinecone.
 */
export async function retrieveListingsVectorless(
  question: string,
  modelOverride?: string,
): Promise<{ listings: RetrievedListing[]; filters: ListingFilters }> {
  const topK = Math.max(1, Number(process.env.RAG_TOP_K || 8) || 8);
  const filters = await reasonTreeFilters(question, modelOverride);
  const listings = loadListingsFromCsv();

  const hasFilters =
    filters.bedroom != null ||
    filters.maxPriceLakh != null ||
    Boolean(filters.sectorQuery) ||
    (filters.sectors?.length ?? 0) > 0;

  const matched = listings
    .filter((listing) =>
      hasFilters ? matchesReasonedFilters(listing, filters) : true,
    )
    .map((listing, index) => {
      let score = 1;
      const sector = listingSector(listing);
      if (filters.sectors?.includes(sector) || sector === filters.sectorQuery) {
        score += 0.2;
      }
      if (filters.bedroom != null && listing.bedroom === filters.bedroom) {
        score += 0.05;
      }
      // Stable-ish ranking: cheaper first within match set when budget set.
      if (filters.maxPriceLakh != null && listing.priceInLakh != null) {
        score += (filters.maxPriceLakh - listing.priceInLakh) / 10_000;
      }
      return toRetrieved(listing, score + index * 1e-9);
    });

  const priceSort = getPriceSort(question);
  matched.sort((a, b) => {
    if (priceSort) {
      const pa = a.priceInLakh ?? -1;
      const pb = b.priceInLakh ?? -1;
      return priceSort === "highest" ? pb - pa : pa - pb;
    }
    return b.score - a.score;
  });

  const answerFilters: ListingFilters = {
    bedroom: filters.bedroom,
    maxPriceLakh: filters.maxPriceLakh,
    sectorQuery: filters.sectorQuery,
  };

  if (!hasFilters) {
    return { listings: matched.slice(0, topK), filters: answerFilters };
  }

  return {
    listings: matched.slice(0, Math.max(topK, 40)),
    filters: answerFilters,
  };
}

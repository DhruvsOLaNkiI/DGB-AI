import { loadEnvConfig } from "@next/env";
import {
  embedTexts,
  getEmbeddingDimensions,
  getEmbeddingModelName,
} from "../src/lib/embeddings";
import { loadListingsFromCsv } from "../src/lib/listings";
import { getPineconeConfig, getPineconeIndex } from "../src/lib/pinecone";
import { listingMetadata } from "../src/lib/rag";

loadEnvConfig(process.cwd());

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryable(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /pinecone|network|timeout|ECONN|fetch failed|503|502|429|unreachable/i.test(
    message,
  );
}

async function upsertWithRetry(
  index: ReturnType<typeof getPineconeIndex>,
  records: Array<{
    id: string;
    values: number[];
    metadata: ReturnType<typeof listingMetadata>;
  }>,
  attempts = 6,
) {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      await index.upsert({ records });
      return;
    } catch (error) {
      lastError = error;
      if (!isRetryable(error) || attempt === attempts) throw error;
      const waitMs = Math.min(30_000, 2000 * attempt);
      console.warn(
        `Pinecone upsert failed (attempt ${attempt}/${attempts}). Waiting ${waitMs / 1000}s...`,
      );
      await sleep(waitMs);
    }
  }
  throw lastError;
}

async function main() {
  const { apiKey: pineconeKey, indexName } = getPineconeConfig();

  if (!pineconeKey) {
    throw new Error("PINECONE_API_KEY is missing in .env.local");
  }

  const embedBatch = Math.max(
    1,
    Number(process.env.EMBED_BATCH_SIZE || 32) || 32,
  );
  const upsertBatch = 100;
  const startOffset = Math.max(
    0,
    Number(process.env.INDEX_START_OFFSET || 0) || 0,
  );

  const listings = loadListingsFromCsv();
  console.log(`Loaded ${listings.length} listings from CSV`);
  console.log(
    `Embedding model: ${getEmbeddingModelName()} (${getEmbeddingDimensions()} dims)`,
  );
  console.log(`Upserting into Pinecone index: ${indexName}`);
  if (startOffset > 0) {
    console.log(`Resuming from offset ${startOffset}`);
  }

  console.log("Loading MiniLM model (first run may download weights)...");
  await embedTexts(["warmup listing embedding"]);
  console.log("Model ready.");

  const index = getPineconeIndex(pineconeKey);
  let upserted = startOffset;

  for (let i = startOffset; i < listings.length; i += embedBatch) {
    const slice = listings.slice(i, i + embedBatch);
    const vectors = await embedTexts(slice.map((listing) => listing.text));

    if (vectors.length !== slice.length) {
      throw new Error(
        `Embedding count mismatch at offset ${i}: got ${vectors.length}, expected ${slice.length}`,
      );
    }

    const records = slice.map((listing, offset) => ({
      id: listing.id,
      values: vectors[offset]!,
      metadata: listingMetadata(listing),
    }));

    for (let j = 0; j < records.length; j += upsertBatch) {
      await upsertWithRetry(index, records.slice(j, j + upsertBatch));
    }

    upserted += slice.length;
    console.log(`Indexed ${upserted}/${listings.length}`);
  }

  console.log("Done. Pinecone index `minilm` is ready for RAG chat.");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});

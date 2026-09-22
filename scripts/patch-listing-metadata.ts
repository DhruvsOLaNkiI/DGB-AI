/**
 * Patch Pinecone metadata with `sector` (no re-embedding).
 * Needed so queries like "3 BHK in sector 75" can filter in Pinecone.
 *
 *   npm run patch-listing-metadata
 */
import { loadEnvConfig } from "@next/env";
import { loadListingsFromCsv } from "../src/lib/listings";
import { getPineconeConfig, getPineconeIndex } from "../src/lib/pinecone";
import { listingMetadata } from "../src/lib/rag";

loadEnvConfig(process.cwd());

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const { apiKey, indexName } = getPineconeConfig();
  if (!apiKey) throw new Error("PINECONE_API_KEY missing");

  const listings = loadListingsFromCsv();
  const index = getPineconeIndex(apiKey);
  const batchSize = 50;

  console.log(`Patching sector metadata on ${indexName} (${listings.length} rows)...`);

  for (let i = 0; i < listings.length; i += batchSize) {
    const slice = listings.slice(i, i + batchSize);
    await Promise.all(
      slice.map((listing) =>
        index.update({
          id: listing.id,
          metadata: listingMetadata(listing),
        }),
      ),
    );
    console.log(`Patched ${Math.min(i + batchSize, listings.length)}/${listings.length}`);
    await sleep(50);
  }

  console.log("Done. Sector filters will work in chat.");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});

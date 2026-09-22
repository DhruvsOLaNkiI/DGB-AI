import { Pinecone } from "@pinecone-database/pinecone";

export function getPineconeConfig() {
  const apiKey = process.env.PINECONE_API_KEY?.trim();
  const indexName = process.env.PINECONE_INDEX?.trim() || "db-ai";
  const host = process.env.PINECONE_HOST?.trim() || undefined;
  const topK = Number(process.env.RAG_TOP_K || 8);

  return {
    apiKey,
    indexName,
    host,
    topK: Number.isFinite(topK) && topK > 0 ? topK : 8,
  };
}

export function createPineconeClient(apiKey: string) {
  return new Pinecone({ apiKey });
}

export function getPineconeIndex(apiKey: string) {
  const { indexName, host } = getPineconeConfig();
  const client = createPineconeClient(apiKey);
  return host ? client.index(indexName, host) : client.index(indexName);
}

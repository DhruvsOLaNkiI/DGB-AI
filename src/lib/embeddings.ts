import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";

const DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_DIMS = 384;

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

export function getEmbeddingModelName() {
  return (
    process.env.LOCAL_EMBEDDING_MODEL?.trim() ||
    process.env.EMBEDDING_MODEL?.trim() ||
    DEFAULT_MODEL
  );
}

export function getEmbeddingDimensions() {
  const dims = Number(
    process.env.EMBEDDING_DIMENSIONS ||
      process.env.GEMINI_EMBEDDING_DIMENSIONS ||
      DEFAULT_DIMS,
  );
  return Number.isFinite(dims) && dims > 0 ? dims : DEFAULT_DIMS;
}

async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    const model = getEmbeddingModelName();
    extractorPromise = pipeline(
      "feature-extraction",
      model,
    ) as Promise<FeatureExtractionPipeline>;
  }
  return extractorPromise;
}

function tensorToVector(output: {
  data: Float32Array | number[];
  tolist?: () => unknown;
}): number[] {
  if (typeof output.tolist === "function") {
    const listed = output.tolist();
    if (Array.isArray(listed)) {
      const flat = Array.isArray(listed[0]) ? (listed[0] as number[]) : listed;
      return flat.map(Number);
    }
  }
  return Array.from(output.data as Float32Array | number[]).map(Number);
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];

  const extractor = await getExtractor();
  const expectedDims = getEmbeddingDimensions();
  const vectors: number[][] = [];

  for (const text of texts) {
    const output = await extractor(text, {
      pooling: "mean",
      normalize: true,
    });

    const values = tensorToVector(
      output as { data: Float32Array | number[]; tolist?: () => unknown },
    );
    if (values.length !== expectedDims) {
      throw new Error(
        `Expected ${expectedDims}-dim embedding, got ${values.length}. Check EMBEDDING_DIMENSIONS and Pinecone index dimension (minilm should be 384).`,
      );
    }
    vectors.push(values);
  }

  return vectors;
}

export async function embedQuery(query: string): Promise<number[]> {
  const [vector] = await embedTexts([query]);
  if (!vector) {
    throw new Error("Failed to embed query.");
  }
  return vector;
}

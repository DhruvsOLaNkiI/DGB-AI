import { NextResponse } from "next/server";
import {
  indexDocumentFile,
  readDocumentRegistry,
  saveUploadedFile,
} from "@/lib/vectorless-docs";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function GET() {
  const documents = await readDocumentRegistry();
  return NextResponse.json({ documents });
}

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "Attach a PDF or CSV as form field `file`." },
        { status: 400 },
      );
    }

    const name = file.name.toLowerCase();
    if (!name.endsWith(".pdf") && !name.endsWith(".csv")) {
      return NextResponse.json(
        {
          error:
            "Upload a .pdf or .csv. (PageIndex indexes PDFs; CSV is converted to PDF first.)",
        },
        { status: 400 },
      );
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    if (buffer.byteLength === 0) {
      return NextResponse.json({ error: "Empty file." }, { status: 400 });
    }
    if (buffer.byteLength > 40 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File too large (max 40MB)." },
        { status: 400 },
      );
    }

    const savedPath = await saveUploadedFile(file.name, buffer);
    const document = await indexDocumentFile(savedPath);
    return NextResponse.json({ ok: true, document });
  } catch (error) {
    let message =
      error instanceof Error ? error.message : "Failed to index document";
    if (/rate.?limit|429|tokens per minute|TPM/i.test(message)) {
      message =
        "Groq rate limit hit while indexing (too many tokens/minute). Wait ~20 seconds and try again, or keep dropdown on “Listings (CSV tree)” — your clean_dataset.csv is already loaded without upload.";
    }
    console.error("[api/vectorless/documents]", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

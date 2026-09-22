import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "csv-parse/sync";

export type ListingRecord = {
  id: string;
  address: string;
  sector: string;
  bedroom: number | null;
  type1: string;
  type2: string;
  priceInLakh: number | null;
  size: number | null;
  carpetArea: number | null;
  rate: number | null;
  status: string;
  furnishing: string;
  typeOfSale: string;
  ageOfProperty: string;
  bathrooms: number | null;
  facing: string;
  amenities: string[];
  /** Exact CSV amenity column → true only when cell is True/1/yes. */
  amenityFlags: Record<string, boolean>;
  text: string;
};

/** CSV amenity columns → human labels used in answers / filters. */
export const AMENITY_COLUMNS: Array<{ column: string; label: string }> = [
  { column: "amenities_sports_facility", label: "sports facility" },
  { column: "amenities_shopping_mall", label: "shopping mall" },
  { column: "amenities_maintenance_staff", label: "maintenance staff" },
  { column: "amenities_jogging_track", label: "jogging track" },
  { column: "amenities_atm", label: "atm" },
  { column: "amenities_gymnasium", label: "gym" },
  { column: "amenities_indoor_games", label: "indoor games" },
  { column: "amenities_rain_water_harvesting", label: "rain water harvesting" },
  { column: "amenities_swimming_pool", label: "swimming pool" },
  { column: "amenities_intercom", label: "intercom" },
  { column: "amenities_cafeteria", label: "cafeteria" },
  { column: "amenities_full_power_backup", label: "full power backup" },
  { column: "amenities_lift", label: "lift" },
  { column: "amenities_childrens_play_area", label: "children's play area" },
  { column: "amenities_car_parking", label: "car parking" },
  { column: "amenities_landscaped_gardens", label: "landscaped gardens" },
  { column: "amenities_24_x_7_security", label: "24x7 security" },
  { column: "amenities_club_house", label: "club house" },
  { column: "amenities_staff_quarter", label: "staff quarter" },
  { column: "amenities_hospital", label: "hospital" },
  { column: "amenities_multipurpose_room", label: "multipurpose room" },
  { column: "amenities_vaastu_compliant", label: "vaastu compliant" },
  { column: "amenities_golf_course", label: "golf course" },
  { column: "amenities_school", label: "school" },
];

export function isAmenityTruthy(value: unknown): boolean {
  const raw = asString(value).toLowerCase();
  return raw === "true" || raw === "1" || raw === "yes";
}

function collectAmenityFlags(row: RawRow): Record<string, boolean> {
  const flags: Record<string, boolean> = {};
  for (const { column } of AMENITY_COLUMNS) {
    // Missing column → false (never treat as matching True).
    flags[column] = isAmenityTruthy(row[column]);
  }
  return flags;
}

function collectAmenities(row: RawRow): string[] {
  return AMENITY_COLUMNS.filter(({ column }) => isAmenityTruthy(row[column])).map(
    ({ label }) => label,
  );
}

/** Normalize "Sector 75, Noida" → "sector 75" for filters / metadata. */
export function extractSector(address: string): string {
  const match = address.toLowerCase().match(/sector\s*([0-9]+[a-z]?)/i);
  return match ? `sector ${match[1]!.toLowerCase()}` : "";
}

type RawRow = Record<string, string>;

function uniqueHeaders(headers: string[]): string[] {
  const seen = new Map<string, number>();
  return headers.map((raw) => {
    const key = String(raw ?? "").trim() || "col";
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);
    return count > 1 ? `${key}_${count}` : key;
  });
}

function asString(value: unknown): string {
  if (value == null) return "";
  return String(value).trim();
}

function asNumber(value: unknown): number | null {
  const raw = asString(value);
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function listingToChunk(listing: Omit<ListingRecord, "text">): string {
  const idNum = listing.id.replace(/^listing-/, "");
  const bhk =
    listing.bedroom != null
      ? `${listing.bedroom} ${listing.type1 || "BHK"}`.trim()
      : listing.type1 || "Property";
  const price =
    listing.priceInLakh != null
      ? `${Number(listing.priceInLakh.toFixed(2))} lakh`
      : "price not listed";
  const sizeBits = [
    listing.size != null ? `${listing.size} sqft` : "",
    listing.carpetArea != null ? `carpet ${listing.carpetArea} sqft` : "",
    listing.rate != null ? `${Math.round(listing.rate)}/sqft` : "",
  ].filter(Boolean);

  const lines = [
    `Property #${idNum}`,
    `Location: ${listing.address || "Noida"}`,
    `${bhk} ${listing.type2 || ""}`.trim(),
    `Price (price_in_lakh): ${price}`,
    sizeBits.length ? `Area: ${sizeBits.join(" | ")}` : "",
    listing.carpetArea != null && listing.carpetArea > 0
      ? `Carpet area (carpet_area): ${listing.carpetArea} sqft`
      : "",
    listing.size != null && listing.size > 0
      ? `Size (size): ${listing.size} sqft`
      : "",
    listing.rate != null && listing.rate > 0
      ? `Rate (rate, ₹ per sqft — not total price): ₹${Math.round(listing.rate).toLocaleString("en-IN")}/sqft`
      : "",
    listing.status ? `Status: ${listing.status}` : "",
    listing.furnishing ? `Furnishing: ${listing.furnishing}` : "",
    listing.typeOfSale ? `Sale type: ${listing.typeOfSale}` : "",
    listing.bathrooms != null ? `Bathrooms: ${listing.bathrooms}` : "",
    listing.facing ? `Facing: ${listing.facing}` : "",
    listing.ageOfProperty
      ? `Age (age_of_property): ${listing.ageOfProperty} years`
      : "",
    listing.amenities.length
      ? `Amenities: ${listing.amenities.join(", ")}`
      : "",
  ].filter(Boolean);

  return lines.join("\n");
}

export function loadListingsFromCsv(csvPath?: string): ListingRecord[] {
  const path = resolve(
    process.cwd(),
    csvPath || process.env.LISTINGS_CSV_PATH || "data/clean_dataset.csv",
  );
  const raw = readFileSync(path, "utf8");
  const rows = parse(raw, {
    columns: (headers: string[]) => uniqueHeaders(headers),
    skip_empty_lines: true,
    relax_column_count: true,
    trim: true,
  }) as RawRow[];

  return rows.map((row, index) => {
    const address = asString(row.address).toLowerCase();
    const base = {
      id: `listing-${index + 1}`,
      address,
      sector: extractSector(address),
      bedroom: asNumber(row.bedroom),
      type1: asString(row.type1),
      type2: asString(row.type2),
      priceInLakh: asNumber(row.price_in_lakh),
      size: asNumber(row.size),
      carpetArea: asNumber(row.carpet_area),
      rate: asNumber(row.rate),
      status: asString(row.status),
      furnishing: asString(row["status.1"]),
      typeOfSale: asString(row.type_of_sale),
      ageOfProperty: asString(row.age_of_property),
      bathrooms: asNumber(row.bathrooms),
      facing: asString(row.facing),
      amenities: collectAmenities(row),
      amenityFlags: collectAmenityFlags(row),
    };

    return {
      ...base,
      text: listingToChunk(base),
    };
  });
}

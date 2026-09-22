/**
 * Shared rules for interpreting Noida listing rows (CSV / PageIndex / LLM).
 * Keep rate (₹/sqft) distinct from price_in_lakh (total price).
 */
export const DATA_INTERPRETATION_RULES = `CRITICAL DATA INTERPRETATION RULES:
1. Raw listing fields may appear pipe-delimited (e.g. size | price_in_lakh | rate). Never confuse per-square-foot rate with total price_in_lakh. rate is already ₹ per sqft; price_in_lakh is the full property price in lakh.
2. When checking numerical conditions (age_of_property, price_in_lakh, bedroom, size, carpet_area, rate), use only the named column. If a value is missing, null, empty, or NaN, say it is not listed — never invent numbers.
3. Multi-condition queries (location + bedrooms + amenities + budget/age, etc.) require EVERY condition to match. Exclude any property that fails even one condition.
4. Always finish complete answers. Do not truncate mid-sentence. When listing properties, write clear human-readable sentences (location, BHK, total price in lakh, size/carpet if asked, rate only if asked or labeled as ₹/sqft). Never reply with raw row indexes alone.
5. BOOLEAN / AMENITY FILTERS: When the user asks for amenities or boolean columns (e.g. swimming pool = True, power backup = True, amenities_swimming_pool), include ONLY rows where those exact columns are True. Exclude False, missing, empty, and null. Never assume every property in a location has the amenity. Never report a count equal to the full location size unless every property truly matches.`;

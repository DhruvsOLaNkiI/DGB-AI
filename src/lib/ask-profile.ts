/** Long-term ASK DGB-SUP preferences (Phase 3) — survives new chats in this browser. */

export const ASK_PROFILE_KEY = "dbg-ai-ask-profile";

export type AskProfile = {
  budget_min_lakh?: number | null;
  budget_max_lakh?: number | null;
  bhk?: number | null;
  sectors?: string[];
  goal?: string | null;
  property_types?: string[];
  horizon_years?: number | null;
  amenities?: string[];
  updatedAt?: number;
};

export function emptyAskProfile(): AskProfile {
  return {
    budget_min_lakh: null,
    budget_max_lakh: null,
    bhk: null,
    sectors: [],
    goal: null,
    property_types: [],
    horizon_years: null,
    amenities: [],
  };
}

export function loadAskProfile(): AskProfile {
  if (typeof window === "undefined") return emptyAskProfile();
  try {
    const raw = window.localStorage.getItem(ASK_PROFILE_KEY);
    if (!raw) return emptyAskProfile();
    const parsed = JSON.parse(raw) as AskProfile;
    return {
      ...emptyAskProfile(),
      ...parsed,
      sectors: Array.isArray(parsed.sectors) ? parsed.sectors : [],
      property_types: Array.isArray(parsed.property_types)
        ? parsed.property_types
        : [],
      amenities: Array.isArray(parsed.amenities) ? parsed.amenities : [],
    };
  } catch {
    return emptyAskProfile();
  }
}

export function saveAskProfile(profile: AskProfile): void {
  if (typeof window === "undefined") return;
  const next: AskProfile = {
    ...emptyAskProfile(),
    ...profile,
    updatedAt: Date.now(),
  };
  window.localStorage.setItem(ASK_PROFILE_KEY, JSON.stringify(next));
}

/** Merge server ask_state into long-term profile (later facts win). */
export function mergeAskProfile(
  base: AskProfile,
  patch: Partial<AskProfile> | null | undefined,
): AskProfile {
  if (!patch) return base;
  const next: AskProfile = { ...base };
  if ("budget_min_lakh" in patch) next.budget_min_lakh = patch.budget_min_lakh ?? null;
  if ("budget_max_lakh" in patch) next.budget_max_lakh = patch.budget_max_lakh ?? null;
  if ("bhk" in patch && patch.bhk != null) next.bhk = patch.bhk;
  if ("goal" in patch && patch.goal) next.goal = patch.goal;
  if ("horizon_years" in patch && patch.horizon_years != null) {
    next.horizon_years = patch.horizon_years;
  }
  if (Array.isArray(patch.sectors) && patch.sectors.length) {
    next.sectors = [...patch.sectors];
  }
  if (Array.isArray(patch.property_types) && patch.property_types.length) {
    next.property_types = [...patch.property_types];
  }
  if (Array.isArray(patch.amenities) && patch.amenities.length) {
    // Accumulate unique amenities over time
    const set = new Set([...(base.amenities || []), ...patch.amenities]);
    next.amenities = [...set];
  }
  next.updatedAt = Date.now();
  return next;
}

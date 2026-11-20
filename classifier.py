"""
RESQ-AI :: Citizen Report Classifier (English / Urdu / Roman Urdu)
-------------------------------------------------------------------
MVP implementation: keyword + rule-based multilingual classifier.

WHY NOT XLM-RoBERTa TODAY
Fine-tuning/downloading a transformer needs internet + a labeled dataset,
neither available in this build session. This module keeps the exact
same output contract a fine-tuned XLM-R classifier would produce, so it's
a drop-in replacement later:

    classify_report(text) -> {
        "language": "en" | "ur" | "roman_ur" | "mixed",
        "category": one of CATEGORIES,
        "severity": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
        "confidence": float,
        "matched_terms": [str, ...]
    }

UPGRADE PATH (documented, not implemented today):
  text -> XLM-RoBERTa tokenizer -> fine-tuned classification head -> label
  Fine-tune on CrisisMMD + your own labeled Urdu/Roman Urdu report set.
"""

from __future__ import annotations
import re

CATEGORIES = [
    "PEOPLE_NEED_RESCUE",
    "MEDICAL_EMERGENCY",
    "FLOOD",
    "ROAD_BLOCKED",
    "BUILDING_DAMAGE",
    "FOOD_WATER_NEEDED",
    "SHELTER_NEEDED",
    "OTHER",
]

# Keyword banks. Roman Urdu spelling varies a lot person to person, so we
# include common variants. This list is deliberately extensible — add
# terms as real citizen reports come in during the build/demo phase.
KEYWORDS = {
    "PEOPLE_NEED_RESCUE": {
        "en": ["trapped", "stuck", "stranded", "rescue", "help us", "can't get out", "surrounded by water"],
        "ur": ["پھنسے", "پھنس گئے", "بچاؤ", "مدد چاہیے", "نکالو"],
        "roman_ur": ["phans", "phanse", "phansay", "bachao", "madad chahiye", "nikalo", "atkay"],
    },
    "MEDICAL_EMERGENCY": {
        "en": ["injured", "bleeding", "unconscious", "sick", "medicine", "pregnant", "labor pain", "heart attack"],
        "ur": ["زخمی", "بیمار", "دوائی", "حاملہ", "بے ہوش"],
        "roman_ur": ["zakhmi", "bemar", "beemar", "dawai", "hamila", "behosh", "dard"],
    },
    "FLOOD": {
        "en": ["flood", "water level", "water rising", "submerged", "overflow"],
        "ur": ["سیلاب", "پانی", "طغیانی"],
        "roman_ur": ["seelab", "pani", "paani", "sailab", "tughyani", "pani bhar gaya"],
    },
    "ROAD_BLOCKED": {
        "en": ["road blocked", "bridge collapsed", "route closed", "can't pass", "road washed away"],
        "ur": ["راستہ بند", "پل ٹوٹ گیا", "سڑک بند"],
        "roman_ur": ["rasta band", "pul toot", "sadak band", "road band"],
    },
    "BUILDING_DAMAGE": {
        "en": ["collapsed", "house destroyed", "wall fell", "roof caved", "building damaged"],
        "ur": ["گھر گر گیا", "چھت گر گئی", "دیوار گری"],
        "roman_ur": ["ghar gir gaya", "chatt gir gayi", "deewar giri", "makan tabah"],
    },
    "FOOD_WATER_NEEDED": {
        "en": ["no food", "no clean water", "hungry", "need food", "need drinking water"],
        "ur": ["کھانا نہیں", "پانی نہیں", "بھوکے"],
        "roman_ur": ["khana nahi", "pani nahi hai peene ka", "bhookay", "khaana chahiye"],
    },
    "SHELTER_NEEDED": {
        "en": ["no shelter", "homeless", "need tent", "nowhere to stay", "camp"],
        "ur": ["پناہ گاہ", "خیمہ", "رہنے کی جگہ نہیں"],
        "roman_ur": ["panah gah", "khaima", "rehne ki jaga nahi"],
    },
}

CRITICAL_MARKERS_EN = ["children", "kids", "elderly", "old", "many people", "can't breathe", "drowning"]
CRITICAL_MARKERS_UR = ["بچے", "بوڑھے", "ڈوب"]
CRITICAL_MARKERS_ROMAN = ["bachon", "bachay", "buddhay", "boorhay", "doob", "dub"]


def _detect_language(text: str) -> str:
    has_urdu_script = bool(re.search(r"[\u0600-\u06FF]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_urdu_script and has_latin:
        return "mixed"
    if has_urdu_script:
        return "ur"
    if has_latin:
        # Heuristic: if latin text contains known roman-urdu tokens, call it roman_ur
        lowered = text.lower()
        roman_hits = sum(
            1 for cat in KEYWORDS.values() for term in cat["roman_ur"] if term in lowered
        )
        return "roman_ur" if roman_hits > 0 else "en"
    return "mixed"


def classify_report(text: str) -> dict:
    if not text or not text.strip():
        return {
            "language": "unknown",
            "category": "OTHER",
            "severity": "LOW",
            "confidence": 0.0,
            "matched_terms": [],
        }

    lowered = text.lower()
    language = _detect_language(text)

    scores = {cat: 0 for cat in CATEGORIES if cat != "OTHER"}
    matched_terms = []

    for cat, lang_map in KEYWORDS.items():
        for lang_key in ("en", "ur", "roman_ur"):
            for term in lang_map[lang_key]:
                needle = term if lang_key == "ur" else term.lower()
                haystack = text if lang_key == "ur" else lowered
                if needle in haystack:
                    scores[cat] += 1
                    matched_terms.append(term)

    best_category = max(scores, key=scores.get) if any(scores.values()) else "OTHER"
    total_hits = sum(scores.values())

    # Severity: rescue/medical always escalate faster; presence of
    # vulnerable-group markers pushes anything to CRITICAL.
    critical_hit = (
        any(m in lowered for m in CRITICAL_MARKERS_EN)
        or any(m in text for m in CRITICAL_MARKERS_UR)
        or any(m in lowered for m in CRITICAL_MARKERS_ROMAN)
    )

    if best_category in ("PEOPLE_NEED_RESCUE", "MEDICAL_EMERGENCY") or critical_hit:
        severity = "CRITICAL"
    elif best_category in ("FLOOD", "ROAD_BLOCKED", "BUILDING_DAMAGE"):
        severity = "HIGH" if total_hits >= 2 else "MODERATE"
    elif best_category in ("FOOD_WATER_NEEDED", "SHELTER_NEEDED"):
        severity = "MODERATE"
    else:
        severity = "LOW"

    # Confidence proxy: more distinct keyword hits -> higher confidence.
    # A fine-tuned classifier would replace this with a real softmax score.
    confidence = min(0.5 + 0.15 * total_hits, 0.95) if total_hits > 0 else 0.35

    return {
        "language": language,
        "category": best_category,
        "severity": severity,
        "confidence": round(confidence, 2),
        "matched_terms": list(dict.fromkeys(matched_terms)),  # dedupe, keep order
    }


if __name__ == "__main__":
    tests = [
        "People are trapped near the bridge, children with them.",
        "لوگ پل کے قریب پھنسے ہوئے ہیں",
        "log pul ke qareeb phansay huay hain, bachay bhi hain",
        "pani gharon mein daakhil ho gaya hai",
        "We are fine, just checking in.",
    ]
    for t in tests:
        print(t, "->", classify_report(t))

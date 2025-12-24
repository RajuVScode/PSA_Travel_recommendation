"""
Outfit planner helper

Provides an LLM instruction template `LLM_INSTRUCTIONS` and a simple
rule-based `recommend_outfits()` function to compute counts for
outdoor, dinner, and activity wears given user activities and preferences.
"""
from typing import List, Dict, Optional

LLM_INSTRUCTIONS = """
You are a planning assistant that recommends the number of outfits required
for a user's upcoming plans. Categorize outfits into:

- Outdoor wears: casual/daytime outings, sightseeing, nature, travel.
- Dinner wears: evening meals, semi-formal/formal dinners, parties.
- Activity wears: workouts, sports, hiking, beach/pool, yoga, high-movement activities.

Base recommendations on the user's activity list (with dates/times if available),
overall trip duration or schedule, the user's stated intention (e.g., minimal packing,
fashion-forward, sustainable reuse), climate and laundry options if provided, and
reasonable reuse logic (e.g., dinner wear can often be reused unless heavily soiled).

Return a JSON object with keys: `outdoor_wears`, `dinner_wears`, `activity_wears`,
`total_wears`, `assumptions` (list of short strings), and `explanation` (human-friendly
summary of how you arrived at the numbers).

Example input schema:
{
  "activities": [
    {"type": "sightseeing", "date": "2025-06-01"},
    {"type": "dinner", "date": "2025-06-01", "time": "19:00"},
    {"type": "hike", "date": "2025-06-02"}
  ],
  "duration_days": 3,
  "intention": "minimal",  // or "fashion", "sustainable", "balanced"
  "climate": "hot",       // or "cold", "temperate"
  "laundry_available": false
}

Guiding heuristics a planner should follow:
- Base outdoor wears: 1 per day.
- Dinner wears: 0.5–1 per evening depending on formality and user intention. Treat explicit dinner/party events as dinner wear needs.
- Activity wears: 1 per high-sweat/activity instance; if multiple activities occur same day, try to reuse (e.g., gym then yoga same day -> 1-2 depending on intensity).
- Reuse adjustments: `minimal` => reduce counts by ~25–40% via reuse; `fashion` => increase variety (+1–2 dinner/outdoor); `sustainable` => encourage reuse (reduce by ~40%); `balanced` => default.
- Laundry reduces needs: if laundry_available is true, reduce overall wears by 30–60% depending on frequency.
- Climate: hot increases activity wear replacement likelihood; cold may allow more reuse for dinner wear.

When implementing programmatically, prefer deterministic, explainable arithmetic rather
than opaque rules. Always include assumptions and a short rationale.
"""


def _classify_activity(name: str) -> str:
    s = (name or "").lower()
    if any(x in s for x in ["dinner", "restaurant", "party", "formal", "gala"]):
        return "dinner"
    if any(x in s for x in ["hike", "trek", "ride", "surf", "swim", "gym", "workout", "yoga", "run", "sport", "beach"]):
        return "activity"
    if any(x in s for x in ["sightseeing", "tour", "city", "walk", "explore", "travel", "visit", "day"]):
        return "outdoor"
    # fallback: consider as outdoor
    return "outdoor"


def recommend_outfits(
    activities: List[Dict],
    duration_days: Optional[int] = None,
    intention: str = "balanced",
    climate: Optional[str] = None,
    laundry_available: bool = False,

    # New optional context parameters allow the function to use weather, events,
    # parsed intent, and LLM prompts when forming recommendations.
    weather: Optional[Dict] = None,
    events: Optional[List] = None,
    parsed_intent: Optional[Dict] = None,
    system_prompt: Optional[str] = None,
    user_message: Optional[str] = None,
) -> Dict:
    """Return a recommendation dict for outfit counts and reasoning.

    activities: list of dicts or strings describing planned activities.
    duration_days: total trip/event days (used when activities don't cover all days).
    intention: one of 'minimal', 'fashion', 'sustainable', 'balanced'.
    climate: optional climate hint: 'hot', 'cold', 'temperate'.
    laundry_available: whether laundry reduces packing needs.
    """

    # Normalize activities list to types and dates
    parsed = []
    for a in activities or []:
        if isinstance(a, str):
            typ = _classify_activity(a)
            parsed.append({"type": typ, "raw": a})
        elif isinstance(a, dict):
            typ = a.get("type") or _classify_activity(a.get("name") or a.get("activity") or "")
            parsed.append({"type": typ, **a})

    # If no explicit activities provided, try to extract from events/context
    if not parsed and events:
        for ev in events:
            if isinstance(ev, dict):
                name = ev.get("name") or ev.get("title") or ev.get("summary") or ev.get("description")
                date = ev.get("date") or ev.get("start") or ev.get("start_date")
                typ = _classify_activity(name or "outing")
                parsed.append({"type": typ, "raw": name, "date": date})
            else:
                typ = _classify_activity(str(ev))
                parsed.append({"type": typ, "raw": str(ev)})

    # Infer climate from provided weather if not explicitly passed
    if climate is None and weather:
        # support a few common keys: temp, temperature, avg_temp, summary, description
        temp = None
        for k in ("avg_temp", "temperature", "temp", "avg_temperature"):
            try:
                if weather.get(k) is not None:
                    temp = float(weather.get(k))
                    break
            except Exception:
                continue
        summary = (str(weather.get("summary") or weather.get("description") or "") ).lower()
        if temp is not None:
            if temp >= 25:
                climate = "hot"
            elif temp <= 10:
                climate = "cold"
            else:
                climate = "temperate"
        elif any(x in summary for x in ["hot", "warm", "heat"]):
            climate = "hot"
        elif any(x in summary for x in ["cold", "snow", "freez", "chill"]):
            climate = "cold"
        else:
            climate = "temperate"

    # Try to detect laundry availability from prompts/context if not provided
    if not laundry_available:
        text_ctx = " ".join([s for s in [system_prompt, user_message] if s])
        if text_ctx and any(x in text_ctx.lower() for x in ["laundry", "washer", "wash", "laundromat", "laundry available"]):
            laundry_available = True

    # Try to detect user intention from prompts if not explicitly provided
    if (not intention or intention == "balanced") and user_message:
        um = user_message.lower()
        if any(x in um for x in ["minimal pack", "minimal", "pack light", "travel light"]):
            intention = "minimal"
        elif any(x in um for x in ["fashion", "outfits", "stylish", "dress up"]):
            intention = "fashion"
        elif any(x in um for x in ["sustain", "sustainable", "reuse", "eco"]):
            intention = "sustainable"

    # Count explicit event days
    days_with_events = set()
    for p in parsed:
        date = p.get("date")
        if date:
            days_with_events.add(date)

    if duration_days is None:
        duration_days = max(1, len(days_with_events)) if days_with_events else 1

    # Base rates
    base_outdoor_per_day = 1
    base_dinner_per_day = 0.5
    base_activity_per_event = 1

    # Tally events
    outdoor_events = sum(1 for p in parsed if p["type"] == "outdoor")
    dinner_events = sum(1 for p in parsed if p["type"] == "dinner")
    activity_events = sum(1 for p in parsed if p["type"] == "activity")

    # Start with defaults
    outdoor_wears = max(outdoor_events, int(round(base_outdoor_per_day * duration_days)))
    dinner_wears = max(dinner_events, int(round(base_dinner_per_day * duration_days)))
    activity_wears = max(activity_events, 0)

    # If activities not daily, ensure at least one outdoor wear per day
    if outdoor_wears < duration_days:
        outdoor_wears = duration_days

    # Adjust for climate: hot -> +10-20% activity apparel (sweat), cold -> dinners more reusable
    if climate == "hot":
        activity_wears = int(round(activity_wears * 1.15))
    elif climate == "cold":
        dinner_wears = max(1, int(round(dinner_wears * 0.9)))

    # Intention adjustments
    intent = (intention or "balanced").lower()
    if intent == "minimal":
        outdoor_wears = max(1, int(round(outdoor_wears * 0.7)))
        dinner_wears = max(0, int(round(dinner_wears * 0.7)))
        activity_wears = max(0, int(round(activity_wears * 0.8)))
    elif intent == "sustainable":
        outdoor_wears = max(1, int(round(outdoor_wears * 0.6)))
        dinner_wears = max(0, int(round(dinner_wears * 0.6)))
        activity_wears = max(0, int(round(activity_wears * 0.7)))
    elif intent == "fashion":
        # add variety: one extra outdoor and one extra dinner if trip >1 day
        if duration_days > 1:
            outdoor_wears += 1
            dinner_wears += 1

    # Laundry availability reduces needs
    if laundry_available:
        outdoor_wears = max(1, int(round(outdoor_wears * 0.6)))
        dinner_wears = max(0, int(round(dinner_wears * 0.6)))
        activity_wears = max(0, int(round(activity_wears * 0.7)))

    # Ensure integers
    outdoor_wears = int(max(0, outdoor_wears))
    dinner_wears = int(max(0, dinner_wears))
    activity_wears = int(max(0, activity_wears))

    total = outdoor_wears + dinner_wears + activity_wears

    assumptions = []
    assumptions.append(f"Assumed {duration_days} day(s) of travel/event.")
    assumptions.append("Outdoor: ~1 per day baseline, reusable when not dirty.")
    assumptions.append("Dinner: counted per explicit evening events; reusable if not soiled.")
    assumptions.append("Activity: one fresh outfit per high-sweat event; may need extras for multi-day activities.")
    if laundry_available:
        assumptions.append("Laundry available mid-trip reduces total needed.")
    if intent in ("minimal", "sustainable"):
        assumptions.append(f"Intent '{intent}' prioritizes reuse and fewer items.")

    explanation_lines = [
        f"Outdoor wears: {outdoor_wears} (base + events)",
        f"Dinner wears: {dinner_wears} (events & formality)",
        f"Activity wears: {activity_wears} (per high-sweat activity)",
        f"Total outfits recommended: {total}",
    ]

    return {
        "outdoor_wears": outdoor_wears,
        "dinner_wears": dinner_wears,
        "activity_wears": activity_wears,
        "total_wears": total,
        "assumptions": assumptions,
        "explanation": "\n".join(explanation_lines),
    }


if __name__ == "__main__":
    # quick interactive demo
    sample = [
        {"type": "sightseeing", "date": "2025-06-01"},
        {"type": "dinner", "date": "2025-06-01", "time": "19:00"},
        {"type": "hike", "date": "2025-06-02"},
    ]
    print(recommend_outfits(sample, duration_days=3, intention="minimal", climate="temperate", laundry_available=False))

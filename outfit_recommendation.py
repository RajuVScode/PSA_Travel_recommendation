
"""
Outfit planner using Azure OpenAI (LLM-driven)

- Provides a system prompt for outfit planning.
- Uses Azure OpenAI function calling to return structured JSON with outfit counts.
- Includes guardrails and minimal fallback in case the LLM fails.

Environment variables:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_KEY
- AZURE_OPENAI_MODEL  (e.g., "gpt-4o-mini")

Usage:
    python outfit_recommendation.py
"""

import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI

# Initialize environment
load_dotenv()

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
)

DEFAULT_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o-mini")


# ----------------------------
# System Prompt (LLM behavior)
# ----------------------------
def build_system_prompt() -> str:
    """
    Build the system prompt guiding the LLM to produce outfit recommendations.
    """
    return (
        "You are a planning assistant that recommends the number of outfits required "
        "for a user's upcoming plans.\n"
        "Categorize outfits into:\n"
        "- Outdoor wears: casual/daytime outings, sightseeing, nature, travel.\n"
        "- Dinner wears: evening meals, semi-formal/formal dinners, parties.\n"
        "- Activity wears: workouts, sports, hiking, beach/pool, yoga, high-movement activities.\n\n"
        "Base recommendations on the user's activity list (with dates/times if available), "
        "overall trip duration or schedule, the user's intention (e.g., minimal packing, "
        "fashion-forward, sustainable reuse), climate and laundry options if provided, and "
        "reasonable reuse logic (e.g., dinner wear can often be reused unless heavily soiled).\n\n"
        "Return a function call with JSON arguments containing:\n"
        "- outfit_counts: { outdoor_wears, dinner_wears, activity_wears, total_wears }\n"
        "- assumptions: list of short strings used in reasoning\n"
        "- explanation: short human-friendly summary\n"
        "- reuse_logic: short sentence (optional)\n"
        "- notes: short packing tip (optional)\n\n"
        "Guiding heuristics:\n"
        "- Outdoor: ~1 per day (adjust for reuse and intention).\n"
        "- Dinner: 0.5–1 per evening depending on formality/intention; explicit dinners need dinner wear.\n"
        "- Activity: 1 per high-sweat instance; light activities can reuse across sessions.\n"
        "- Reuse adjustments: minimal => reduce ~25–40%; fashion => increase variety +1–2; "
        "  sustainable => reduce ~40%; balanced => default.\n"
        "- Laundry: if available, reduce overall wears by 30–60% depending on frequency.\n"
        "- Climate: hot increases activity wear replacement; cold allows more reuse for dinner/outdoor.\n\n"
        "Always include assumptions and a short rationale. Keep counts integers and non-negative."
    )


# ---------------------------------
# Tool (Function) schema for output
# ---------------------------------
def outfit_tool_schema() -> Dict[str, Any]:
    """
    Define function-calling schema to force the model to return structured JSON.
    """
    return {
        "type": "function",
        "function": {
            "name": "return_outfit_recommendation",
            "description": "Return outfit counts and reasoning for the user's plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outfit_counts": {
                        "type": "object",
                        "properties": {
                            "outdoor_wears": {"type": "integer", "minimum": 0},
                            "dinner_wears": {"type": "integer", "minimum": 0},
                            "activity_wears": {"type": "integer", "minimum": 0},
                            "total_wears": {"type": "integer", "minimum": 0},
                        },
                        "required": ["outdoor_wears", "dinner_wears", "activity_wears"],
                        "additionalProperties": False,
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "explanation": {"type": "string"},
                    "reuse_logic": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["outfit_counts", "assumptions", "explanation"],
                "additionalProperties": False,
            },
        },
    }


# -------------------
# Input normalization
# -------------------
def _normalize_activities(activities: List[Dict]) -> List[Dict]:
    """
    Normalize activities to a consistent form.
    Each item becomes: { type: 'outdoor|dinner|activity', name?, date?, time? }
    If 'type' missing, try inferring from 'name'/'activity'.
    """
    def _classify(name: str) -> str:
        s = (name or "").lower()
        if any(x in s for x in ["dinner", "restaurant", "party", "formal", "gala"]):
            return "dinner"
        if any(x in s for x in ["hike", "trek", "ride", "surf", "swim", "gym", "workout", "yoga", "run", "sport", "beach"]):
            return "activity"
        if any(x in s for x in ["sightseeing", "tour", "city", "walk", "explore", "travel", "visit", "day"]):
            return "outdoor"
        return "outdoor"

    parsed: List[Dict] = []
    for a in activities or []:
        if isinstance(a, str):
            parsed.append({"type": _classify(a), "name": a})
        elif isinstance(a, dict):
            name = a.get("name") or a.get("activity") or a.get("type") or ""
            typ = (a.get("type") or _classify(name))
            item = {"type": typ}
            for k in ("name", "date", "time", "intensity"):
                if a.get(k) is not None:
                    item[k] = a.get(k)
            parsed.append(item)
    return parsed


def _build_user_payload(
    activities: List[Dict],
    duration_days: Optional[int],
    intention: str,
    climate: Optional[str],
    laundry_available: bool,
    weather: Optional[Dict],
    destination: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    extra_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Prepare a compact JSON payload sent to the LLM.
    """
    return {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
        "intention": intention,
        "climate": climate,
        "laundry_available": bool(laundry_available),
        "weather": weather or {},
        "activities": _normalize_activities(activities),
        "context": extra_context or {},
    }


# --------------------------
# LLM call + result handling
# --------------------------
def _post_validate(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guardrails: ensure integers, non-negative, compute total if missing, limit assumptions.
    """
    if not isinstance(result, dict):
        return {
            "outfit_counts": {"outdoor_wears": 0, "dinner_wears": 0, "activity_wears": 0, "total_wears": 0},
            "assumptions": ["Fallback applied due to invalid LLM output."],
            "explanation": "Defaulted to zeros; please retry.",
        }
    oc = result.get("outfit_counts") or {}
    ow = max(int(oc.get("outdoor_wears") or 0), 0)
    dw = max(int(oc.get("dinner_wears") or 0), 0)
    aw = max(int(oc.get("activity_wears") or 0), 0)
    total = int(oc.get("total_wears") or (ow + dw + aw))
    total = max(total, ow + dw + aw)

    result["outfit_counts"] = {
        "outdoor_wears": ow,
        "dinner_wears": dw,
        "activity_wears": aw,
        "total_wears": total,
    }

    # prune/normalize assumptions
    assumptions = result.get("assumptions") or []
    assumptions = [str(a).strip() for a in assumptions if str(a).strip()]
    # Cap excessive assumptions for UX
    if len(assumptions) > 10:
        assumptions = assumptions[:10]
    result["assumptions"] = assumptions

    # explanation defaults
    if not result.get("explanation"):
        result["explanation"] = (
            "Counts derived from activities, duration, intention, climate, and laundry settings with reasonable reuse."
        )

    return result


def recommend_outfits_llm(
    activities: List[Dict],
    duration_days: Optional[int] = None,
    intention: str = "balanced",
    climate: Optional[str] = None,
    laundry_available: bool = False,
    weather: Optional[Dict] = None,
    destination: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    extra_context: Optional[Dict] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI to compute outfit recommendations.

    Returns:
        dict with keys: outfit_counts, assumptions, explanation, (optional reuse_logic, notes)
    """
    model = model or DEFAULT_MODEL
    system_prompt = build_system_prompt()
    payload = _build_user_payload(
        activities=activities,
        duration_days=duration_days,
        intention=intention,
        climate=climate,
        laundry_available=laundry_available,
        weather=weather,
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        extra_context=extra_context,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_prompt},
                # Provide the payload as a JSON string for clarity and semantic parsing
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tools=[outfit_tool_schema()],
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        # Prefer function/tool call structured output
        if getattr(message, "tool_calls", None):
            # Take the first tool call
            tool_call = message.tool_calls[0]
            args_json = tool_call.function.arguments
            result = json.loads(args_json)
            return _post_validate(result)

        # If no tool call, try to parse content directly (JSON fallback)
        content = message.content
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content)
        else:
            text = content or ""
        result = json.loads(text)
        return _post_validate(result)

    except Exception as e:
        # Minimal arithmetic fallback: use duration and activity counts
        print(f"[DEBUG] LLM recommendation failed: {e}")

        # Fallback heuristic
        parsed = _normalize_activities(activities)
        outdoor_events = sum(1 for p in parsed if p["type"] == "outdoor")
        dinner_events = sum(1 for p in parsed if p["type"] == "dinner")
        activity_events = sum(1 for p in parsed if p["type"] == "activity")
        days = duration_days or max(1, len({p.get('date') for p in parsed if p.get('date')})) or 1

        base_outdoor = max(outdoor_events, days)  # ~1/day
        base_dinner = max(dinner_events, round(0.6 * days))  # 0.5–1 per evening
        base_activity = max(activity_events, 0)

        # Intention adjustment (simple fallback)
        intent = (intention or "balanced").lower()
        if intent == "minimal":
            base_outdoor = round(base_outdoor * 0.7)
            base_dinner = round(base_dinner * 0.7)
            base_activity = round(base_activity * 0.8)
        elif intent == "fashion":
            base_outdoor += 1
            base_dinner += 1
        elif intent == "sustainable":
            base_outdoor = round(base_outdoor * 0.6)
            base_dinner = round(base_dinner * 0.6)
            base_activity = round(base_activity * 0.7)

        if laundry_available:
            base_outdoor = max(round(base_outdoor * 0.6), 0)
            base_dinner = max(round(base_dinner * 0.7), 0)
            base_activity = max(round(base_activity * 0.7), 0)

        total = max(base_outdoor, 0) + max(base_dinner, 0) + max(base_activity, 0)

        return {
            "outfit_counts": {
                "outdoor_wears": max(base_outdoor, 0),
                "dinner_wears": max(base_dinner, 0),
                "activity_wears": max(base_activity, 0),
                "total_wears": total,
            },
            "assumptions": [
                "Fallback arithmetic used due to LLM error",
                f"Days={days}, OutdoorEvents={outdoor_events}, DinnerEvents={dinner_events}, ActivityEvents={activity_events}",
                f"Intention={intention}, Laundry={laundry_available}, Climate={climate}",
            ],
            "explanation": "Computed via simple heuristics as a backup. Try again for LLM-driven reasoning.",
        }


# -----------------
# Example execution
# -----------------
if __name__ == "__main__":
    example_activities = [
        {"type": "sightseeing", "date": "2025-06-01"},
        {"type": "dinner", "date": "2025-06-01", "time": "19:00"},
        {"type": "hike", "date": "2025-06-02", "intensity": "high"},
        {"name": "gym", "date": "2025-06-03", "intensity": "high"},
    ]
    result = recommend_outfits_llm(
        activities=example_activities,
        duration_days=3,
        intention="minimal",         # "minimal" | "fashion" | "sustainable" | "balanced"
        climate="hot",               # optional: "hot" | "cold" | "temperate"
        laundry_available=False,
        weather={"avg_temp": 32, "summary": "hot and humid"},
        destination="Pondicherry",
        start_date="2025-06-01",
        end_date="2025-06-03",
        extra_context={"notes": "carry quick-dry tees"},
    )
    print(json.dumps(result, indent=2))

import os
import json
import requests
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
# Initialize environment
load_dotenv()

# ---- IMPORTANT: provide your own geocode_location(destination) implementation or keep the one you already have ----
# Expected return format: {"latitude": <float>, "longitude": <float>, "name": "<city>", "admin1": "<state/region>"}
# def geocode_location(destination: str) -> dict:
#     ...

def _azure_client():
    """
    Create an Azure OpenAI client using environment variables.
    """
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-10-01-preview",  # update if needed
    )

def _prompt_forecast_json(location_name, lat, lon, data_source, start_date, end_date,
                          avg_high, avg_low, precip_chance, total_precip, predominant_code):
    """
    Prompt for LLM to return the final JSON response that matches your schema.
    The LLM is instructed to use numbers EXACTLY as provided and only generate descriptive strings/notes.
    """
    # We use a compact, explicit instruction to minimize hallucination and preserve values.
    user_prompt = f"""
You are a weather assistant. Return EXACTLY one JSON object with this schema:

- destination: string
- latitude: number
- longitude: number
- avg_high_c: number
- avg_low_c: number
- precipitation_chance: number
- precipitation_mm: number
- wind_info: string
- daylight_hours: number
- weather_description: string
- notes: string
- data_source: string (one of: "forecast", "historical")
- error: null or string

CONSTRAINTS:
- Use the numeric values provided below EXACTLY. Do not recompute them.
- Map the weather code (WMO-like) to a human readable description. If unknown, use a sensible general description.
- Keep daylight_hours at 12.0 unless you have strong reason otherwise.
- For wind_info, summarize likely winds succinctly (e.g., "Calm", "Light", "Moderate", "Strong") based on typical conditions;
  keep it conservative if uncertain.
- Notes should state the data source and the location context succinctly.
- If any field is missing or invalid, return error string (but only if truly necessary).

INPUTS:
destination: "{location_name}"
latitude: {lat}
longitude: {lon}
data_source: "{data_source}"
date_range: "{start_date or 'N/A'} to {end_date or start_date or 'N/A'}"

Provided numeric values (use as-is):
- avg_high_c: {avg_high if avg_high is not None else 'null'}
- avg_low_c: {avg_low if avg_low is not None else 'null'}
- precipitation_chance: {precip_chance if precip_chance is not None else 0}
- precipitation_mm: {total_precip if total_precip is not None else 0}
- predominant_weather_code: {predominant_code if predominant_code is not None else 'null'}

Return ONLY the JSON object; no extra text.
"""
    return user_prompt

def geocode_location(destination: str, timeout: int = 5) -> dict:
    """
    Geocode using Open-Meteo's free geocoding endpoint.
    Returns dict with keys: latitude, longitude, name (display), admin1 (region)
    """
    if not destination:
        return None
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={destination}&count=1"
        geo_response = requests.get(geo_url, timeout=timeout).json()
        if not geo_response.get("results"):
            return None
        geo = geo_response["results"][0]
        return {
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "name": geo.get("name"),
            "admin1": geo.get("admin1"),
        }
    except Exception:
        return None
    
def _prompt_climatology_json(destination, month_name):
    """
    Prompt to ask LLM for an estimated climatology when the real-time data is unavailable.
    This removes hardcoded climatology tables.
    """
    user_prompt = f"""
You are a weather assistant. Return EXACTLY one JSON object with this schema:

- destination: string
- latitude: null
- longitude: null
- avg_high_c: number
- avg_low_c: number
- precipitation_chance: number
- precipitation_mm: number
- wind_info: string
- daylight_hours: number
- weather_description: string
- notes: string
- data_source: "climatology"
- error: null

TASK:
- Provide typical climatology estimates for "{destination}" in the month "{month_name}" using your general knowledge.
- Be reasonable and conservative. If the location is ambiguous, assume a typical temperate scenario and note the limitation.
- Set daylight_hours to 12.0.
- Precipitation chance is an estimated percentage; precipitation_mm is a rough monthly-total estimate.
- Keep wind_info succinct (e.g., "Light", "Moderate", "Moderate to strong") and consistent with the season.

Return ONLY the JSON object; no extra text.
"""
    return user_prompt


def get_weather_context(destination: str, start_date: str = None, end_date: str = None) -> dict:
    """
    Fetch weather data for a destination and date range, then delegate final JSON formatting
    (including descriptions and notes) to Azure OpenAI (LLM). Falls back to LLM-driven climatology
    if the real-time forecast/archive is unavailable.

    The returned dict structure matches the original function's keys.
    """
    if not destination:
        return {
            "destination": None,
            "error": "No destination provided",
            "data_source": "error",
        }

    # Geocode destination
    geo = geocode_location(destination)
    if not geo:
        return {
            "destination": destination,
            "error": f"Location '{destination}' not found or geocoding failed",
            "data_source": "error",
        }

    lat, lon = geo.get("latitude"), geo.get("longitude")
    location_name = f"{geo.get('name', '')}{',' + geo.get('admin1', '') if geo.get('admin1') else ''}".strip(",")

    # Determine forecast vs historical
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.now()

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        if start_dt.date() > today.date():
            # Use forecast
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={(start_date or start_dt.date().isoformat())}"
                f"&end_date={(end_date or start_dt.date().isoformat())}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
            )
            response = requests.get(url, timeout=10).json()
            data_source = "forecast"
        else:
            # Use archive
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={(start_date or start_dt.date().isoformat())}"
                f"&end_date={(end_date or start_dt.date().isoformat())}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
            )
            response = requests.get(url, timeout=10).json()
            data_source = "historical"
    except Exception as e:
        # If API fails, fall back to LLM climatology
        print(f"[DEBUG] Weather API call failed: {e}")
        response = None
        data_source = "error"

    client = _azure_client()
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-4o-mini" or "gpt-4.1"

    # If we have daily data, compute metrics here (deterministic), then ask LLM to produce final JSON.
    if response and "daily" in response:
        daily = response.get("daily", {})
        temps_max = daily.get("temperature_2m_max", []) or []
        temps_min = daily.get("temperature_2m_min", []) or []
        precip = daily.get("precipitation_sum", []) or []
        codes = daily.get("weathercode", []) or []

        avg_high = round(sum(temps_max) / len(temps_max), 1) if temps_max else None
        avg_low = round(sum(temps_min) / len(temps_min), 1) if temps_min else None
        total_precip = round(sum(precip), 1) if precip else 0.0
        precip_chance = round((sum(1 for p in precip if p > 0) / len(precip) * 100), 1) if precip else 0.0
        predominant_code = codes[0] if codes else None

        prompt = _prompt_forecast_json(
            location_name=location_name,
            lat=lat,
            lon=lon,
            data_source=data_source,
            start_date=start_date,
            end_date=end_date,
            avg_high=avg_high,
            avg_low=avg_low,
            precip_chance=precip_chance,
            total_precip=total_precip,
            predominant_code=predominant_code,
        )

        # Request strict JSON output (chat.completions supports json_object; Responses API supports json_schema)
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You generate structured weather context JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            # If decoding fails, build a minimal fallback without hardcoded tables
            obj = {
                "destination": location_name,
                "latitude": lat,
                "longitude": lon,
                "avg_high_c": avg_high,
                "avg_low_c": avg_low,
                "precipitation_chance": precip_chance,
                "precipitation_mm": total_precip,
                "wind_info": "Moderate",  # conservative default wording
                "daylight_hours": 12.0,
                "weather_description": "Variable conditions",
                "notes": f"Based on {data_source} data for {location_name}.",
                "data_source": data_source,
                "error": None,
            }

        # Ensure required keys exist and normalize types
        obj.setdefault("destination", location_name)
        obj.setdefault("latitude", lat)
        obj.setdefault("longitude", lon)
        obj.setdefault("data_source", data_source)
        obj.setdefault("daylight_hours", 12.0)
        obj.setdefault("error", None)
        print(f"[DEBUG get_weather_context] Result: {obj}")
        return obj

    # Otherwise, LLM-only climatology (no hardcoded month table)
    print(f"[DEBUG] Falling back to LLM climatology for {destination}")
    month_name = None
    if start_date:
        try:
            month_name = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B")
        except Exception:
            month_name = "Unknown"

    prompt = _prompt_climatology_json(destination=destination, month_name=month_name or "Unknown")
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You generate structured weather context JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Minimal fallback if LLM returns malformed JSON
        obj = {
            "destination": destination,
            "latitude": None,
            "longitude": None,
            "avg_high_c": None,
            "avg_low_c": None,
            "precipitation_chance": None,
            "precipitation_mm": None,
            "wind_info": "Moderate",
            "daylight_hours": 12.0,
            "weather_description": "Typical seasonal conditions",
            "notes": f"Using LLM estimated climatology for {month_name or 'the destination'}. Real-time forecast unavailable.",
            "data_source": "climatology",
            "error": None,
        }

    # Normalize required fields
    obj.setdefault("destination", destination)
    obj.setdefault("latitude", None)
    obj.setdefault("longitude", None)
    obj.setdefault("data_source", "climatology")
    obj.setdefault("daylight_hours", 12.0)
    obj.setdefault("error", None)
    print(f"[DEBUG get_weather_context] Result: {obj}")
    return obj

if __name__ == "__main__":
    get_destionation = input("Enter your travel destination: ")
    get_start_date = input("Enter start date (YYYY-MM-DD) or leave blank: ")
    get_end_date = input("Enter end date (YYYY-MM-DD) or leave blank: ")
    
    weather = get_weather_context(get_destionation, get_start_date, get_end_date)
    print(weather)
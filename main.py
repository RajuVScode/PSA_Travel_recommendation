"""
Travel Recommendation Pipeline
Integrates RAG (Azure AI Search), Weather APIs, and LLM to generate formal travel plans.
"""

import os
import re
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from parse_travel_intent import extract_with_llm

# Initialize environment
load_dotenv()

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
)



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


# =====================================================================
# 2. WEATHER: Fetch weather data
# =====================================================================
def get_weather_context(destination: str, start_date: str = None, end_date: str = None) -> dict:
    """
    Fetch weather data for a destination and date range.
    Falls back to climatology if real-time forecast unavailable.
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
    location_name = f"{geo.get('name', '')}{',' + geo.get('admin1', '') if geo.get('admin1') else ''}"
    
    # Determine forecast vs historical
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.now()
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Use forecast if start_date is in the future, otherwise archive
        if start_dt.date() > today.date():
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date or start_dt.date().isoformat()}&end_date={end_date or start_dt.date().isoformat()}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
            response = requests.get(url, timeout=10).json()
            data_source = "forecast"
        else:
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date or start_dt.date().isoformat()}&end_date={end_date or start_dt.date().isoformat()}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
            response = requests.get(url, timeout=10).json()
            data_source = "historical"
    except Exception as e:
        print(f"[DEBUG] Weather API call failed: {e}")
        data_source = "error"
        response = None
    
    # Parse response
    if response and "daily" in response:
        daily = response.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes = daily.get("weathercode", [])
        
        avg_high = round(sum(temps_max) / len(temps_max), 1) if temps_max else None
        avg_low = round(sum(temps_min) / len(temps_min), 1) if temps_min else None
        total_precip = round(sum(precip), 1) if precip else 0
        precip_chance = round((sum(1 for p in precip if p > 0) / len(precip) * 100), 1) if precip else 0
        
        # Map weather code to description
        code_desc = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "foggy with rime", 51: "drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
            61: "rain", 63: "moderate rain", 65: "heavy rain",
            71: "snow", 73: "moderate snow", 75: "heavy snow",
            80: "rain showers", 81: "moderate showers", 82: "violent showers",
            95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
        }
        predominant_code = codes[0] if codes else None
        weather_desc = code_desc.get(predominant_code, "variable conditions")
        
        daylight_hours = 12.0
        
        notes = f"Based on {data_source} data for {location_name}."
        
        return {
            "destination": location_name,
            "latitude": lat,
            "longitude": lon,
            "avg_high_c": avg_high,
            "avg_low_c": avg_low,
            "precipitation_chance": precip_chance,
            "precipitation_mm": total_precip,
            "wind_info": "Moderate winds",
            "daylight_hours": daylight_hours,
            "weather_description": weather_desc,
            "notes": notes,
            "data_source": data_source,
            "error": None,
        }
    else:
        # Fallback to climatology
        print(f"[DEBUG] Falling back to climatology for {destination}")
        climatology = {
            "april": {"avg_high": 12, "avg_low": 6, "precip_chance": 40, "wind": "moderate"},
            "may": {"avg_high": 17, "avg_low": 10, "precip_chance": 35, "wind": "light to moderate"},
            "june": {"avg_high": 22, "avg_low": 14, "precip_chance": 30, "wind": "light"},
            "july": {"avg_high": 24, "avg_low": 16, "precip_chance": 25, "wind": "light"},
            "august": {"avg_high": 23, "avg_low": 15, "precip_chance": 28, "wind": "light"},
            "september": {"avg_high": 19, "avg_low": 12, "precip_chance": 35, "wind": "light to moderate"},
            "october": {"avg_high": 14, "avg_low": 9, "precip_chance": 45, "wind": "moderate"},
            "november": {"avg_high": 10, "avg_low": 6, "precip_chance": 50, "wind": "moderate"},
            "december": {"avg_high": 7, "avg_low": 3, "precip_chance": 55, "wind": "moderate to strong"},
            "january": {"avg_high": 6, "avg_low": 2, "precip_chance": 60, "wind": "moderate to strong"},
            "february": {"avg_high": 6, "avg_low": 2, "precip_chance": 55, "wind": "moderate"},
            "march": {"avg_high": 9, "avg_low": 4, "precip_chance": 50, "wind": "moderate"},
        }
        
        month = None
        if start_date:
            try:
                month = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B").lower()
            except Exception:
                pass
        
        seasonal = climatology.get(month, climatology["april"])
        
        return {
            "destination": destination,
            "latitude": None,
            "longitude": None,
            "avg_high_c": seasonal["avg_high"],
            "avg_low_c": seasonal["avg_low"],
            "precipitation_chance": seasonal["precip_chance"],
            "wind_info": seasonal["wind"],
            "daylight_hours": 12.0,
            "notes": f"Using typical climatology for {month or 'the destination'}. Real-time forecast unavailable.",
            "data_source": "climatology",
            "error": None,
        }
# =====================================================================
# RAW RAG: Schema-agnostic retrieval of full documents (no hardcoded fields)
# =====================================================================


def get_raw_products_from_rag(
    destination: str = None,
    season_hint: str = None,
    weather: dict = None,
    user_prompt: str = None,
    recommend_outfits: str = None,
    top_k: int = 5,
) -> list:
    """
    Perform semantic (then simple) search and return the raw document payloads.
    This function is schema-agnostic: it preserves all keys/values found in each
    retrieved document and performs type-aware normalization to ensure JSON
    serializability. No field names are hardcoded or assumed.

    CHANGELOG:
    - Combines all parameters into a single composite query for similarity search.
    - Sends exactly one search request (semantic, then simple fallback).
    """

    # --- Init Azure Search client ---
    try:
        client = SearchClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            index_name=os.environ.get("AZURE_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY")),
        )
    except Exception as e:
        print(f"[DEBUG RAG RAW] Azure Search client init failed: {e}")
        return []

    # --- Helper: normalize values for JSON ---
    def _normalize_value(v):
        # Recursively convert values to JSON-serializable Python primitives
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, dict):
            return {str(k): _normalize_value(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_normalize_value(i) for i in v]
        # Fallback: string representation
        try:
            return str(v)
        except Exception:
            return None

    # --- Helper: document to normalized dict ---
    def _doc_to_dict(result_obj):
        # Extract the document payload and normalize recursively
        doc = None
        try:
            if hasattr(result_obj, "document"):
                doc = result_obj.document
            elif isinstance(result_obj, dict):
                doc = result_obj
            elif hasattr(result_obj, "to_dict"):
                doc = result_obj.to_dict()
            else:
                doc = json.loads(
                    json.dumps(result_obj, default=lambda o: getattr(o, "__dict__", str(o)))
                )
        except Exception:
            try:
                doc = json.loads(
                    json.dumps(result_obj, default=lambda o: getattr(o, "__dict__", str(o)))
                )
            except Exception:
                doc = {}

        if not isinstance(doc, dict):
            return {}

        normalized = {}
        for k, val in doc.items():
            try:
                normalized[str(k)] = _normalize_value(val)
            except Exception:
                normalized[str(k)] = None
        return normalized

    # --- Build a single composite query string ---
    def _build_composite_query():
        parts = []

        if destination:
            parts.append(f"destination: {destination}")

        if season_hint:
            parts.append(f"season: {season_hint}")

        # Weather: convert dict into readable attributes
        if isinstance(weather, dict) and weather:
            w_parts = []
            # include common keys if present
            if weather.get("avg_high_c") is not None:
                w_parts.append(f"avg_high_c: {weather.get('avg_high_c')} C")
            if weather.get("avg_low_c") is not None:
                w_parts.append(f"avg_low_c: {weather.get('avg_low_c')} C")
            if weather.get("precipitation_chance") is not None:
                w_parts.append(f"precipitation_chance: {weather.get('precipitation_chance')}%")
            if weather.get("condition"):
                w_parts.append(f"condition: {weather.get('condition')}")
            if w_parts:
                parts.append("weather: " + ", ".join(w_parts))

        if recommend_outfits:
            parts.append(f"recommend_outfits: {recommend_outfits}")

        if user_prompt:
            # Keep the natural language as-is to maximize semantic match
            parts.append(f"user_prompt: {user_prompt}")

        # Fallback catch-all context (kept from your original code)
        parts.append("context: any products for travel; store details based on destination and weather")

        # Use separators to help the semantic model parse intent and attributes
        composite = " | ".join(parts)
        return composite

    composite_query = _build_composite_query()
    print(f"[DEBUG RAG RAW] Composite query:\n{composite_query}")

    raw_docs = []

    # --- Single search attempt with composite query ---
    def _try_search(query_text: str):
        try:
            print(f"\n[DEBUG RAG RAW] Trying semantic search for query: '{query_text}'")
            results = client.search(
                search_text=query_text,
                query_type="semantic",
                semantic_configuration_name=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIG"),
                top=top_k,
            )
            res_list = list(results)
            print(f"[DEBUG RAG RAW] Semantic returned {len(res_list)} items")
            if res_list:
                return res_list
        except Exception as e:
            print(f"[DEBUG RAG RAW] Semantic search error: {e}")

        try:
            print(f"[DEBUG RAG RAW] Trying simple search for query: '{query_text}'")
            results = client.search(
                search_text=query_text,
                query_type="simple",
                top=top_k,
            )
            res_list = list(results)
            print(f"[DEBUG RAG RAW] Simple returned {len(res_list)} items")
            return res_list
        except Exception as e2:
            print(f"[DEBUG RAG RAW] Simple search error: {e2}")
            return []

    results = _try_search(composite_query)
    if results:
        for r in results[:top_k]:
            doc = _doc_to_dict(r)
            raw_docs.append(doc)

    if not raw_docs:
        print("[DEBUG RAG RAW] No products found (raw).")

    print(f"[DEBUG RAG RAW] Retrieved {len(raw_docs)} raw documents")
    return raw_docs

    


# =====================================================================
# 4. LLM CONTEXT: Build structured input
# =====================================================================
def build_llm_context(
    user_prompt: str,
    parsed_intent: dict,
    weather: dict,
    products: list,
    events: list = None,
    outfit_recommendations: dict = None,
) -> tuple:    
    """
    Build system and user message for LLM.
    """
    system_prompt = """You are a formal travel and lifestyle recommender.

Given the user's travel prompt, destination, dates, weather context, and top 5 product recommendations, produce a structured, formal response that MUST include:

1) **Weather Overview** – temperatures (high/low), precipitation likelihood, wind, daylight, and seasonal notes
2) **Recommended Activities** – indoor/outdoor options justified by the weather and destination
3) **Clothing/Shoes/Accessories Preferences** – weather-aware and practical recommendations grounded in the forecast
4) **Itinerary** (Weekly or Daily) – when travel-related, a day-by-day or week-by-week plan with suggested activities and attire
5) **Local Events Summary** – Provide a concise summary of relevant local events (title, dates, venue, URL), indicate whether each event is likely outdoor or indoor, and mark events as `weather-sensitive` when appropriate. For each event, comment on how the event's conditions (outdoor/indoor and likely weather) should influence activities, itinerary choices, and product recommendations from the RAG results. If no events are available, state 'No events found for the requested dates. 
6) **Outfit/Product Recommendations** – specific items from the product catalog that align with the weather and activities
7) **Recommended Products from Catalog** – Include specific product information like long_description, family_name, material, available_colors, available_sizes, names, brands, prices, and availability. Generate a formal response that answers the user query using only the product catalog data from RAG results above. 
Do not add any generic product details or assumptions. 
If a field is missing, explicitly mention "Information not available." 
Maintain a professional and formal tone.

Incorporate the 5 recommended products from the RAG to support the plan.

If data is missing, state your assumptions clearly and proceed with best-practice recommendations.

Ensure the tone is formal, professional, and complete. All sections must be present."""
    
   

    # Include weather as-is (schema-agnostic) and products as provided. Products are expected to be JSON-serializable dicts.
    weather_json = weather or {}    
    products_json = products or []
    
    user_message = f"""
User's Travel Request:
{user_prompt}

Parsed Travel Intent:
- Destination: {parsed_intent.get('destination') or 'Not specified'}
- Start Date: {parsed_intent.get('start_date') or 'Not specified'}
- End Date: {parsed_intent.get('end_date') or 'Not specified'}
- Duration: {parsed_intent.get('duration_days') or '?'} days
- Season: {parsed_intent.get('season_hint') or 'Not specified'}

Weather Context:
{json.dumps(weather_json, indent=2)}

Local Events (near destination & dates):
{json.dumps(events or [], indent=2)}

Outfit/Product Recommendations:
{json.dumps(outfit_recommendations, indent=2)}

Top 5 Product Recommendations from Catalog:
{json.dumps(products_json, indent=2)}

Please provide a comprehensive, formal travel plan incorporating the above information. Ensure you include all five required sections, and weave the product details (names, brands, prices) naturally into your recommendations.
"""
    
    return system_prompt, user_message


# =====================================================================
# 5. LLM CALL: Generate formal plan
# =====================================================================
def generate_formal_plan(system_prompt: str, user_message: str) -> str:
    """
    Call the LLM and return formatted response.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating plan: {e}"

# =====================================================================
# 3. EVENTS: Fetch local events from Ticketmaster

def get_local_events(destination: str, start_date: str = None, end_date: str = None, radius_km: int = 50) -> list:
    """
    Fetch local events for a destination and date range using Ticketmaster Discovery API.
    Requires environment variable `TICKETMASTER_KEY`. If not provided, returns [].

    Returns a list of dicts with keys: title, start, end, venue, url, description, weather_sensitive
    """
    api_key = os.environ.get("TICKETMASTER_KEY")
    if not destination:
        print(f"[DEBUG EVENTS] No destination provided to get_local_events()")
        return []
    if not api_key:
        print(f"[DEBUG EVENTS] TICKETMASTER_KEY not set in environment; returning no events")
        return []

    geo = geocode_location(destination)
    if not geo:
        print(f"[DEBUG EVENTS] Geocoding failed for destination: {destination}")
        return []

    lat, lon = geo.get("latitude"), geo.get("longitude")
    if not lat or not lon:
        return []

    # Build date-time range in ISO with time suffix if provided
    def to_dt_iso(d):
        if not d:
            return None
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%dT00:00:00Z")
        except Exception:
            return None

    start_dt = to_dt_iso(start_date)
    end_dt = to_dt_iso(end_date) or start_dt

    params = {
        "apikey": api_key,
        "latlong": f"{lat},{lon}",
        "radius": int(radius_km * 0.621371),  # Ticketmaster uses miles
        "unit": "miles",
        "size": 20,
    }
    if start_dt:
        params["startDateTime"] = start_dt
    if end_dt:
        # make end time end of day
        params["endDateTime"] = end_dt.replace("T00:00:00Z", "T23:59:59Z")

    # Determine SSL verify parameter: allow custom CA bundle via env or default True
    verify_param = True
    trusted_ca = os.environ.get("TICKETMASTER_TRUSTED_CA") or os.environ.get("REQUESTS_CA_BUNDLE")
    if trusted_ca:
        verify_param = trusted_ca

    try:
        print(f"[DEBUG EVENTS] Ticketmaster request params: {params}")
        print(f"[DEBUG EVENTS] Using SSL verify: {verify_param}")
        resp = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params=params,
            timeout=8,
            verify=verify_param,
        )
        print(f"[DEBUG EVENTS] Ticketmaster HTTP {resp.status_code}")
        try:
            data = resp.json()
        except Exception as je:
            print(f"[DEBUG EVENTS] Failed to parse JSON from Ticketmaster response: {je}")
            print(f"[DEBUG EVENTS] Response text (truncated): {resp.text[:1000]}")
            return []
        # If the API returns an error structure, log it
        if not data:
            print("[DEBUG EVENTS] Empty JSON payload from Ticketmaster")
            return []
        if data.get("errors"):
            print(f"[DEBUG EVENTS] Ticketmaster API errors: {data.get('errors')}")
            return []
    except requests.exceptions.SSLError as ssle:
        print(f"[DEBUG EVENTS] SSL error when contacting Ticketmaster: {ssle}")
        # Retry with verification disabled as a last resort (insecure) and inform user
        try:
            print("[DEBUG EVENTS] Retrying Ticketmaster request with SSL verification disabled (insecure)")
            resp = requests.get(
                "https://app.ticketmaster.com/discovery/v2/events.json",
                params=params,
                timeout=8,
                verify=False,
            )
            print(f"[DEBUG EVENTS] Retry HTTP {resp.status_code}")
            try:
                data = resp.json()
            except Exception as je:
                print(f"[DEBUG EVENTS] Failed to parse JSON on retry: {je}")
                print(f"[DEBUG EVENTS] Response text (truncated): {resp.text[:1000]}")
                return []
        except Exception as e:
            print(f"[DEBUG EVENTS] HTTP retry to Ticketmaster failed: {e}")
            return []
    except Exception as e:
        print(f"[DEBUG EVENTS] HTTP request to Ticketmaster failed: {e}")
        return []

    events = []
    embedded = data.get("_embedded", {})
    evs = embedded.get("events", []) if embedded else []
    print(f"[DEBUG EVENTS] Ticketmaster returned {len(evs)} events in payload")
    for e in evs:
        title = e.get("name")
        url = e.get("url")
        dates = e.get("dates", {}).get("start", {})
        start = dates.get("localDate") or dates.get("dateTime")
        end = e.get("dates", {}).get("end", {}).get("localDate") if e.get("dates", {}).get("end") else None
        desc = e.get("info") or e.get("pleaseNote") or ""
        venue = None
        try:
            venue_data = e.get("_embedded", {}).get("venues", [])[0]
            venue = {
                "name": venue_data.get("name"),
                "address": ", ".join(filter(None, [venue_data.get("address", {}).get("line1"), venue_data.get("city", {}).get("name") if venue_data.get("city") else None]))
            }
        except Exception:
            venue = None

        # crude outdoor detection
        name_desc = (title or "") + " " + (desc or "")
        outdoor_keywords = ["festival", "fair", "outdoor", "park", "beach", "street", "market"]
        weather_sensitive = any(k in name_desc.lower() for k in outdoor_keywords)

        events.append({
            "title": title,
            "start": start,
            "end": end,
            "venue": venue,
            "url": url,
            "description": desc,
            "weather_sensitive": weather_sensitive,
        })

    return events


# =====================================================================
def generate_travel_recommendation(user_prompt: str) -> str:
    """
    End-to-end pipeline: parse intent → fetch weather → retrieve products → generate plan.
    """
    print("\n" + "=" * 80)
    print("TRAVEL RECOMMENDATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Parse intent
    print("\n[STEP 1] Parsing travel intent...")    
    parsed_intent = extract_with_llm(user_prompt)
    
    # Step 2: Fetch weather
    print("\n[STEP 2] Fetching weather context...")
    weather = get_weather_context(
        destination=parsed_intent.get("destination"),
        start_date=parsed_intent.get("start_date"),
        end_date=parsed_intent.get("end_date"),
    )
    if weather.get("error"):
        print(f"[WARNING] Weather fetch: {weather['error']}")

    # Step 3: Retrieve local events for the destination/dates
    print("\n[STEP 3] Retrieving local events for destination and dates...")
    events = get_local_events(
        destination=parsed_intent.get("destination"),
        start_date=parsed_intent.get("start_date"),
        end_date=parsed_intent.get("end_date"),
    )
    print(f"[INFO] Retrieved {len(events)} events")    

    # step 4 Append outfit recommendations from outfit_planner (if available)
    
    from outfit_planner import recommend_outfits
    print("\n[STEP 4] Build simple activities list from retrieved `events` and user prompt...")
    
    activities_list = []
    for ev in events or []:
        if isinstance(ev, dict):
            name = ev.get("name") or ev.get("title") or ev.get("summary") or ev.get("description")
            date = ev.get("date") or ev.get("start") or ev.get("start_date")
            activities_list.append({"type": name or "outing", "date": date})
        else:
            activities_list.append(str(ev))

    # Fallback: detect explicit mentions in the user's prompt
    up = (user_prompt or "").lower()
    if not activities_list:
        if "dinner" in up or "party" in up:
            activities_list.append({"type": "dinner"})
        if any(x in up for x in ["hike", "hiking", "gym", "yoga", "run", "beach", "swim"]):
            activities_list.append({"type": "hike"})

    outfit_recommendations = recommend_outfits(
        activities=activities_list,
        duration_days=parsed_intent.get("duration_days"),
        intention="balanced",
        climate=None,
        laundry_available=False,
        weather=weather,
        events=events,
        parsed_intent=parsed_intent
    )       
    print(outfit_recommendations)
    print(f"[INFO] Retrieved {outfit_recommendations}")       
    
    # Step 5: Retrieve products from RAG
    print("\n[STEP 5] Retrieving products from RAG...")
    products = get_raw_products_from_rag(
        destination=parsed_intent.get("destination"),
        season_hint=parsed_intent.get("season_hint"),
        weather=weather,
        user_prompt=user_prompt,
        recommend_outfits=outfit_recommendations,
        top_k=5,
    )
    print(products)
    print(f"[INFO] Retrieved {len(products)} products")   
    
    
    # Step 6: Build LLM context
    print("\n[STEP 6] Building LLM context...")
    system_prompt, user_message = build_llm_context(
        user_prompt=user_prompt,
        parsed_intent=parsed_intent,
        weather=weather,
        products=products,
        events=events,
        outfit_recommendations=outfit_recommendations
    )
    
    # Step 5: Generate plan
    print("\n[STEP 5] Calling LLM to generate formal plan...")
    plan = generate_formal_plan(system_prompt, user_message)

    final_output = f"{plan}"

    
    
    print("\n" + "=" * 80)
    print("FINAL PLAN")
    print("=" * 80)
    
    return final_output


# =====================================================================
# MAIN: Demo Runner
# =====================================================================
if __name__ == "__main__":
    user_input = input("Enter your travel request: ")

    plan = generate_travel_recommendation(user_input)
    print("\n" + plan)
    print("\n" + "=" * 80)

    # Outfit suggestion integration (safe import)
    try:
        from outfit_planner import recommend_outfits

        intent = extract_with_llm(user_input)
        duration = intent.get("duration_days") or 1

        # Build a minimal activities list from parsed intent; if none, rely on duration
        activities = []
        # If the user mentioned dinner or landmark words in the prompt, we might add a dinner event
        if "dinner" in user_input.lower() or "party" in user_input.lower():
            activities.append({"type": "dinner"})

        rec = recommend_outfits(activities, duration_days=duration, intention="balanced", climate=None, laundry_available=False)
        print("\nOutfit suggestions:")
        print(rec.get("explanation"))
        print("Assumptions:")
        for a in rec.get("assumptions", []):
            print(" - ", a)
    except Exception as e:
        print(f"Outfit planner not available: {e}")

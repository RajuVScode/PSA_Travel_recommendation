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


# Helper: recursively search nested documents for likely fields
def _recursive_find_value(doc, keywords):
    """Recursively search dict/list for the first value whose key matches any keyword substring."""
    if doc is None:
        return None
    if isinstance(doc, dict):
        for k, v in doc.items():
            kl = k.lower() if isinstance(k, str) else ""
            for kw in keywords:
                if kw in kl and v:
                    if isinstance(v, (str, int, float)):
                        return v
                    if isinstance(v, list) and v:
                        return v[0]
            found = _recursive_find_value(v, keywords)
            if found:
                return found
    elif isinstance(doc, list):
        for item in doc:
            found = _recursive_find_value(item, keywords)
            if found:
                return found
    return None


def _aggressive_format_from_doc(doc, i):
    """Aggressively extract title, brand, price, sizes, colors from possibly nested doc."""
    # Title
    title = None
    for field in ["display_name", "name", "product_name", "title", "product_title", "item_name"]:
        if doc.get(field):
            title = str(doc.get(field)).strip()
            break
    if not title:
        found = _recursive_find_value(doc, ["title", "name", "product", "display"])
        if found:
            title = str(found)
    if not title:
        title = f"Product {i}"

    # Brand
    brand = None
    for field in ["brand", "manufacturer", "vendor", "brand_name"]:
        if doc.get(field):
            brand = str(doc.get(field)).strip()
            break
    if not brand:
        found = _recursive_find_value(doc, ["brand", "maker", "vendor"])
        if found:
            brand = str(found)
    if not brand:
        brand = "N/A"

    # Category
    category = None
    for field in ["sub_category", "category", "product_category", "product_type", "subcategory"]:
        if doc.get(field):
            category = str(doc.get(field)).strip()
            break
    if not category:
        found = _recursive_find_value(doc, ["category", "type", "subcategory"])
        if found:
            category = str(found)
    if not category:
        category = "General"

    # Price
    price = None
    for price_field in ["web_price", "price", "promo_price", "retail_price", "selling_price", "list_price"]:
        val = doc.get(price_field)
        if val:
            try:
                price_val = float(val)
                if price_val > 100:
                    price = f"${price_val/100:.2f}"
                else:
                    price = f"${price_val:.2f}"
                break
            except (ValueError, TypeError):
                continue
    if not price:
        found = _recursive_find_value(doc, ["price", "amount", "cost"])
        if found:
            try:
                pv = float(found)
                price = f"${pv:.2f}" if pv <= 100 else f"${pv/100:.2f}"
            except Exception:
                price = str(found)
    if not price:
        price = "Price on request"

    # Availability
    availability = "In stock"
    active_status = doc.get("active") or doc.get("in_stock") or doc.get("is_available") or _recursive_find_value(doc, ["in_stock", "available", "availability"])
    if active_status is False or active_status == "false" or active_status == 0:
        availability = "Out of stock"
    elif active_status == "limited":
        availability = "Limited stock"

    # Features
    features = []
    mat = doc.get("material") or _recursive_find_value(doc, ["material", "fabric", "composition"])
    if mat:
        features.append(f"Material: {mat}")

    sizes = doc.get("available_sizes") or _recursive_find_value(doc, ["size", "sizes", "available_sizes"])
    if sizes:
        if isinstance(sizes, list):
            sizes_str = ", ".join(str(s) for s in sizes if s)
        else:
            sizes_str = str(sizes)
        if sizes_str:
            features.append(f"Available sizes: {sizes_str}")

    colors = doc.get("available_colors") or _recursive_find_value(doc, ["color", "colour", "available_colors"])
    if colors:
        if isinstance(colors, list):
            colors_str = ", ".join(str(c) for c in colors if c)
        else:
            colors_str = str(colors)
        if colors_str:
            features.append(f"Available colors: {colors_str}")

    if not features:
        for desc_field in ["short_description", "long_description", "description"]:
            desc = doc.get(desc_field) or _recursive_find_value(doc, ["description", "desc", "summary"]) 
            if desc:
                snippet = str(desc)[:150].strip()
                if snippet:
                    features.append(snippet)
                    break
    if not features:
        features = ["Detailed product specs available"]

    tags = []
    full_text = (str(doc.get("long_description", "")) + " " + str(doc.get("short_description", "")) + " " + str(doc.get("display_name", "")) + " " + str(category)).lower()
    if any(x in full_text for x in ["water", "rain", "proof", "waterproof", "rainproof"]):
        tags.append("waterproof")
    if any(x in full_text for x in ["breathe", "light", "airy", "breathable", "lightweight"]):
        tags.append("breathable")
    if any(x in full_text for x in ["insulate", "warm", "thermal", "fleece", "down", "winter"]):
        tags.append("insulated")
    if "shoes" in category.lower() or "footwear" in full_text:
        tags.append("footwear")
    if "clothing" in category.lower() or "apparel" in full_text:
        tags.append("clothing")
    if not tags:
        tags = ["travel-essential"]

    return {
        "id": doc.get("id") or f"rag_{i}",
        "title": title,
        "brand": brand,
        "category": category,
        "tags": list(set(tags)),
        "features": features,
        "price": str(price),
        "availability": availability,
        "url": doc.get("url") or doc.get("product_url") or _recursive_find_value(doc, ["url", "link", "product_url"]) or "",
        "imageUrl": doc.get("image_url") or doc.get("image") or _recursive_find_value(doc, ["image", "image_url", "photo"]) or "",
    }


# =====================================================================
# 1. PARSING: Extract travel intent from user prompt
# =====================================================================
def parse_travel_intent(user_prompt: str) -> dict:
    """
    Extract destination, start date, end date, duration, and season hint from user prompt.
    """
    destination = None
    start_date = None
    end_date = None
    duration_days = None
    season_hint = None
    
    prompt = user_prompt.lower()

    # Handle single-day keywords: today, tomorrow, yesterday
    single_day_match = re.search(r"\b(today|tomorrow|yesterday)\b", prompt)
    if single_day_match and not start_date:
        kw = single_day_match.group(1).lower()
        today_date = datetime.now().date()
        if kw == "today":
            start_dt = today_date
        elif kw == "tomorrow":
            start_dt = today_date + timedelta(days=1)
        else:  # yesterday
            start_dt = today_date - timedelta(days=1)

        start_date = start_dt.isoformat()
        # By default a single-day mention means a 1-day trip unless duration specified
        if duration_days and not end_date:
            end_dt = start_dt + timedelta(days=duration_days - 1)
            end_date = end_dt.isoformat()
        elif not end_date:
            end_date = start_date
    
    # Try to extract destination (city/country patterns)
    dest_patterns = [
        r"(?:to|in|visiting|travelling to|traveling to|trip to)\s+([A-Za-z\s]+?)(?:\s+for|\s+in|\s+from|$|[,;.])",
        r"(?:destination|going to|heading to)\s+([A-Za-z\s]+?)(?:\s+for|\s+in|\s+from|$|[,;.])",
        r"^([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\s+(?:next|this|last|in|for)",  # Pattern: "Miami Nextweek", "Paris Next month", etc.
        r"\b([A-Z][a-z]{3,})\s+(?:next|this|last|in|for)",  # Capitalized city at start of relative time phrase
    ]
    for pattern in dest_patterns:
        m = re.search(pattern, user_prompt)  # Use original case for better matching
        if m:
            destination = m.group(1).strip().title()
            break
    
    # Try LLM-assisted destination extraction if regex fails
    if not destination:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract the travel destination (city or country) from the prompt. Return JSON: {\"destination\": \"<location>\" or null}"},
                    {"role": "user", "content": f"Prompt: {user_prompt}"}
                ],
                temperature=0.0,
                max_tokens=50,
            )
            result = json.loads(response.choices[0].message.content)
            destination = result.get("destination")
        except Exception as e:
            print(f"[DEBUG] LLM destination extraction failed: {e}")
            destination = None

    # Normalize/clean destination text: remove leading verbs and trailing time phrases
    if destination:
        # remove common leading verbs like 'visit', 'visiting', 'going to', etc.
        destination = re.sub(r'^(visit(?:ing)?|going to|go to|trip to|heading to|destination)\s+', '', destination, flags=re.IGNORECASE)
        # remove trailing relative time phrases like 'next weekend', 'this month'
        destination = re.sub(r'\b(next|this|last)\s+(week|weekend|month|year)\b', '', destination, flags=re.IGNORECASE)
        # strip punctuation and extra whitespace
        destination = destination.strip().strip(' ,.;')
        if destination:
            destination = destination.title()
        else:
            destination = None
    
    # Extract duration (e.g., "2 weeks", "10 days", "3 nights")
    duration_match = re.search(r"(\d+)\s+(week|day|night)s?", prompt)
    if duration_match:
        amount = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        if "week" in unit:
            duration_days = amount * 7
        elif "day" in unit:
            duration_days = amount
        elif "night" in unit:
            duration_days = amount + 1
    
    # Handle relative date references like "next week", "this weekend", etc.
    relative_date_match = re.search(r"(next|this|last)\s+(week|weekend|month)", prompt, re.IGNORECASE)
    if relative_date_match and not start_date:
        ref_type = relative_date_match.group(1).lower()
        ref_period = relative_date_match.group(2).lower()
        today = datetime.now()
        
        if "week" in ref_period:
            if "next" in ref_type:
                # Next week starts from Monday (7 days from now)
                days_until_next_monday = (7 - today.weekday()) % 7
                if days_until_next_monday == 0:
                    days_until_next_monday = 7
                start_dt = today + timedelta(days=days_until_next_monday)
                end_dt = start_dt + timedelta(days=6)
                duration_days = 7
            elif "this" in ref_type:
                # This week
                days_until_monday = -today.weekday()
                start_dt = today + timedelta(days=days_until_monday)
                end_dt = start_dt + timedelta(days=6)
                duration_days = 7
            start_date = start_dt.date().isoformat()
            end_date = end_dt.date().isoformat()
        elif "weekend" in ref_period:
            if "next" in ref_type:
                # Next weekend (Saturday and Sunday)
                days_until_saturday = (5 - today.weekday()) % 7
                if days_until_saturday == 0:
                    days_until_saturday = 7
                start_dt = today + timedelta(days=days_until_saturday)
                end_dt = start_dt + timedelta(days=1)
            else:
                # This weekend
                days_until_saturday = (5 - today.weekday()) % 7
                start_dt = today + timedelta(days=days_until_saturday)
                end_dt = start_dt + timedelta(days=1)
            start_date = start_dt.date().isoformat()
            end_date = end_dt.date().isoformat()
            duration_days = 2
    
    # Try to extract explicit dates
    date_patterns = [
        r"(\d{4}-\d{2}-\d{2})",  # 2025-04-15
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",  # 15/04/2025 or 04-15-2025
        r"(?:from|on)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?)",  # April 15 2025
    ]
    for pattern in date_patterns:
        dates = re.findall(pattern, prompt)
        if dates:
            for date_str in dates:
                try:
                    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"]:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            if start_date is None:
                                start_date = dt.date().isoformat()
                            else:
                                end_date = dt.date().isoformat()
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
    
    # Extract month/season hints
    month_patterns = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    months = re.findall(month_patterns, prompt, flags=re.IGNORECASE)
    if months:
        season_hint = months[0].title()
    
    # If duration is known but dates aren't, compute end_date from start_date
    if start_date and duration_days and not end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=duration_days - 1)
        end_date = end_dt.date().isoformat()
    
    # If no explicit start_date, try to infer from month and year
    if not start_date and season_hint:
        year_match = re.search(r"\b(202\d|203\d)\b", prompt)
        year = int(year_match.group(1)) if year_match else datetime.now().year
        try:
            month_num = datetime.strptime(season_hint, "%B").month
            start_dt = datetime(year, month_num, 1)
            start_date = start_dt.date().isoformat()
            
            if duration_days:
                end_dt = start_dt + timedelta(days=duration_days - 1)
            else:
                if month_num == 12:
                    end_dt = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_dt = datetime(year, month_num + 1, 1) - timedelta(days=1)
            end_date = end_dt.date().isoformat()
            duration_days = (end_dt - start_dt).days + 1
        except Exception as e:
            print(f"[DEBUG] Date inference failed: {e}")
    
    print(f"\n[DEBUG parseTravelIntent]")
    print(f"  Destination: {destination}")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Duration Days: {duration_days}")
    print(f"  Season Hint: {season_hint}")
    
    return {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
        "season_hint": season_hint,
        "raw_prompt": user_prompt,
    }


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
5) **Local Events Summary** – Include specific event information like title, dates, venue, URL, description, and weather sensitivity. Comment on how each event's conditions should influence activities, itinerary choices, and product recommendations. 
6) **Outfit/Product Recommendations** – specific items from the product catalog that align with the weather and activities
7) **Recommended Products from Catalog** – Include specific product information like long_description, family_name, material, available_colors, available_sizes, names, brands, prices, and availability. Generate a formal response that answers the user query using only the product catalog data from RAG results above. 
Do not add any generic product details or assumptions. 
If a field is missing, explicitly mention "Information not available." 
Maintain a professional and formal tone.

Incorporate the 5 recommended products from the RAG to support the plan.

If data is missing, state your assumptions clearly and proceed with best-practice recommendations.

Ensure the tone is formal, professional, and complete. All sections must be present."""
    
     # Events requirement: instruct the model to summarize events and combine with weather/RAG
    system_prompt += "\n\n6) **Local Events Summary** – Provide a concise summary of relevant local events (title, dates, venue, URL), indicate whether each event is likely outdoor or indoor, and mark events as `weather-sensitive` when appropriate. For each event, comment on how the event's conditions (outdoor/indoor and likely weather) should influence activities, itinerary choices, and product recommendations from the RAG results. If no events are available, state 'No events found for the requested dates.'"
    

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
# HELPER: Format product catalog
# =====================================================================
def format_product_catalog(products: list) -> str:
    """
    Format RAG products into a generic, schema-agnostic catalog.
    This renderer does not assume or rename any field; it enumerates
    each document's keys and values as provided.
    """
    if not products:
        return "\n---\nAvailable Products: none found."

    lines = ["\n---\n", "Recommended Products from Catalog:\n"]
    for idx, prod in enumerate(products[:5], 1):
        lines.append(f"\nProduct {idx}:")
        if not isinstance(prod, dict) or not prod:
            lines.append("- (no fields available)")
            continue
        # Sort keys for stable output, but preserve original names
        for k in sorted(prod.keys()):
            v = prod.get(k)
            try:
                # Use compact JSON for complex types
                if isinstance(v, (dict, list)):
                    vstr = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    vstr = "<absent>"
                else:
                    vstr = str(v)
            except Exception:
                vstr = str(v)
            lines.append(f"- {k}: {vstr}")

    lines.append("\n---\n")
    return "\n".join(lines)


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
    parsed_intent = parse_travel_intent(user_prompt)
    
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

        intent = parse_travel_intent(user_input)
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

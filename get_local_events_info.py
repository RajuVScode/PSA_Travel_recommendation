import os
import json
import requests
from datetime import datetime
from get_weather_info import geocode_location

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


if __name__ == "__main__":
    get_destionation = input("Enter your travel destination: ")
    get_start_date = input("Enter start date (YYYY-MM-DD) or leave blank: ")
    get_end_date = input("Enter end date (YYYY-MM-DD) or leave blank: ")
    
    events = get_local_events(get_destionation, get_start_date, get_end_date)
    print(events)
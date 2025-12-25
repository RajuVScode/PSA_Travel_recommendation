import os
import json
from datetime import datetime, timedelta
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
)

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

if __name__ == "__main__":
    get_destionation = input("Enter your travel destination: ")    
    sesson_hint = "winter"  # Example static season hint; replace with actual logic if needed
    
    products = get_raw_products_from_rag(get_destionation, sesson_hint, weather={'destination': 'Miami,Florida', 'latitude': 25.77427, 'longitude': -80.19366, 'avg_high_c': 24.9, 'avg_low_c': 15.0, 'precipitation_chance': 0.0, 'precipitation_mm': 0.0, 'wind_info': 'Calm', 'daylight_hours': 12.0, 'weather_description': 'Clear', 'notes': 'Forecast data for Miami, Florida.', 'data_source': 'forecast', 'error': None},
        user_prompt="travelling to Miami for a beach vacation",
        recommend_outfits="1 outfit per day",
        top_k=5)
    print(products)
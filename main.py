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
from outfit_recommendation import recommend_outfits_llm
from get_weather_info import get_weather_context
from get_local_events_info import get_local_events
from get_products_from_RAG import get_raw_products_from_rag
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, END 

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

# =====================================================================
# ---------------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------------

class TravelState(TypedDict, total=False):
    """Shared state across nodes."""
    user_prompt: str

    # Intermediate artifacts
    parsed_intent: Dict[str, Any]
    weather: Dict[str, Any]
    events: List[Dict[str, Any]]
    activities: List[Dict[str, Any]]
    outfit_recommendations: Any
    products: List[Dict[str, Any]]
    llm_context: Dict[str, Any]

    # Final
    plan: str

    # Logging info (optional)
    logs: List[str]


# ---------------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------------

def node_parse_intent(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n" + "=" * 80)
    logs.append("TRAVEL RECOMMENDATION PIPELINE")
    logs.append("=" * 80)
    logs.append("\n[STEP 1] Parsing travel intent...")

    user_prompt = state["user_prompt"]
    parsed_intent = extract_with_llm(user_prompt)

    logs.append(f"[INFO] Parsed intent: {parsed_intent}")
    return {
        "parsed_intent": parsed_intent,
        "logs": logs,
    }


def node_fetch_weather(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 2] Fetching weather context...")

    pi = state["parsed_intent"]
    weather = get_weather_context(
        destination=pi.get("destination"),
        start_date=pi.get("start_date"),
        end_date=pi.get("end_date"),
    )

    if weather.get("error"):
        logs.append(f"[WARNING] Weather fetch: {weather['error']}")
    else:
        logs.append("[INFO] Weather context fetched successfully.")

    return {
        "weather": weather,
        "logs": logs,
    }


def node_retrieve_events(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 3] Retrieving local events for destination and dates...")

    pi = state["parsed_intent"]
    events = get_local_events(
        destination=pi.get("destination"),
        start_date=pi.get("start_date"),
        end_date=pi.get("end_date"),
    ) or []

    logs.append(f"[INFO] Retrieved {len(events)} events")
    return {
        "events": events,
        "logs": logs,
    }


def node_build_activities_and_outfits(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 4] Build simple activities list from retrieved `events` and user prompt...")

    events = state.get("events", [])
    user_prompt = state["user_prompt"]
    pi = state["parsed_intent"]
    weather = state.get("weather")

    # Build activities list
    activities_list: List[Dict[str, Any]] = []
    for ev in events or []:
        if isinstance(ev, dict):
            name = ev.get("name") or ev.get("title") or ev.get("summary") or ev.get("description")
            date = ev.get("date") or ev.get("start") or ev.get("start_date")
            activities_list.append({"type": name or "outing", "date": date})
        else:
            activities_list.append({"type": str(ev)})

    # Fallback: detect explicit mentions in the user's prompt
    up = (user_prompt or "").lower()
    if not activities_list:
        if "dinner" in up or "party" in up:
            activities_list.append({"type": "dinner"})
        if any(x in up for x in ["hike", "hiking", "gym", "yoga", "run", "beach", "swim"]):
            activities_list.append({"type": "hike"})

    # Outfits via LLM
    outfit_recommendations = recommend_outfits_llm(
        activities=activities_list,
        duration_days=pi.get("duration_days"),
        intention="balanced",
        climate=None,
        laundry_available=False,
        weather=weather,
        destination=pi.get("destination"),
        start_date=pi.get("start_date"),
        end_date=pi.get("end_date"),
        extra_context=events
    )

    logs.append(f"[INFO] Outfit recommendations ready.")
    return {
        "activities": activities_list,
        "outfit_recommendations": outfit_recommendations,
        "logs": logs,
    }


def node_retrieve_products(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 5] Retrieving products from RAG...")

    pi = state["parsed_intent"]
    weather = state.get("weather")
    outfit_recommendations = state.get("outfit_recommendations")

    products = get_raw_products_from_rag(
        destination=pi.get("destination"),
        season_hint=pi.get("season_hint"),
        weather=weather,
        user_prompt=state["user_prompt"],
        recommend_outfits=outfit_recommendations,
        top_k=5,
    ) or []

    logs.append(f"[INFO] Retrieved {len(products)} products")
    return {
        "products": products,
        "logs": logs,
    }


def node_build_llm_context(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 6] Building LLM context...")

    system_prompt, user_message = build_llm_context(
        user_prompt=state["user_prompt"],
        parsed_intent=state["parsed_intent"],
        weather=state.get("weather"),
        products=state.get("products"),
        events=state.get("events"),
        outfit_recommendations=state.get("outfit_recommendations")
    )

    logs.append("[INFO] LLM context built.")
    return {
        "llm_context": {
            "system_prompt": system_prompt,
            "user_message": user_message
        },
        "logs": logs,
    }


def node_generate_plan(state: TravelState) -> TravelState:
    logs = state.get("logs", [])
    logs.append("\n[STEP 7] Calling LLM to generate formal plan...")

    ctx = state["llm_context"]
    plan = generate_formal_plan(ctx["system_prompt"], ctx["user_message"])

    logs.append("\n" + "=" * 80)
    logs.append("FINAL PLAN")
    logs.append("=" * 80)
    # You can log a short preview or entire plan; here we keep just a note.
    logs.append("[INFO] Plan generated.")

    return {
        "plan": plan,
        "logs": logs,
    }


# ---------------------------------------------------------------------------------
# Build the sequential graph
# ---------------------------------------------------------------------------------

def build_travel_graph():
    graph = StateGraph(TravelState)

    # Add nodes
    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("fetch_weather", node_fetch_weather)
    graph.add_node("retrieve_events", node_retrieve_events)
    graph.add_node("activities_and_outfits", node_build_activities_and_outfits)
    graph.add_node("retrieve_products", node_retrieve_products)
    graph.add_node("build_llm_context", node_build_llm_context)
    graph.add_node("generate_plan", node_generate_plan)

    # Set entry point and connect edges in sequence
    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "fetch_weather")
    graph.add_edge("fetch_weather", "retrieve_events")
    graph.add_edge("retrieve_events", "activities_and_outfits")
    graph.add_edge("activities_and_outfits", "retrieve_products")
    graph.add_edge("retrieve_products", "build_llm_context")
    graph.add_edge("build_llm_context", "generate_plan")
    graph.add_edge("generate_plan", END)

    return graph.compile()


# ---------------------------------------------------------------------------------
# Convenience wrapper (drop-in replacement)
# ---------------------------------------------------------------------------------

def generate_travel_recommendation_langgraph(user_prompt: str) -> str:
    """
    End-to-end pipeline executed as a sequential LangGraph.
    Mirrors your original behavior with structured nodes.
    """
    app = build_travel_graph()

    initial_state: TravelState = {
        "user_prompt": user_prompt,
        "logs": []
    }

    final_state = app.invoke(initial_state)

    # If you wish to inspect logs:
    for line in final_state.get("logs", []):
        print(line)

    return final_state.get("plan", "")
# ====================================================================

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
# MAIN: Demo Runner
# =====================================================================
if __name__ == "__main__":
    user_input = input("Enter your travel request: ")
    plan = generate_travel_recommendation_langgraph(user_input)
    print("\n" + plan)
    print("\n" + "=" * 80)
    

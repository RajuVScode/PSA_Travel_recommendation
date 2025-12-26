
"""
Travel Intent Clarifier Chatbot using Gradio + Azure OpenAI

Features:
- Collects destination, travel date (single or range), activities, preferred brand, clothes, budget.
- Asks one clarifying question at a time until all fields are filled or confirmed.
- Uses Azure OpenAI (Chat Completions) with strict JSON output for reliability.
- Displays a live summary of the user's current plan.
"""

import os
import json
import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

import gradio as gr
from dotenv import load_dotenv

# Azure OpenAI SDK (OpenAI Python 1.x)
# pip install openai
from openai import AzureOpenAI


# ------------------------
# Environment Setup
# ------------------------
load_dotenv()  # Loads .env if present

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()


if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
    raise RuntimeError(
        "Missing Azure OpenAI configuration. Please set: "
        "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
        "(optionally AZURE_OPENAI_API_VERSION)."
    )

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
)


# ------------------------
# Intent Data Model
# ------------------------
@dataclass
class TravelIntent:
    destination: Optional[str] = None
    travel_date: Optional[str] = None  # Allow "YYYY-MM-DD" or "YYYY-MM-DD to YYYY-MM-DD"
    activities: List[str] = field(default_factory=list)
    preferred_brand: Optional[str] = None
    clothes: Optional[str] = None
    budget_amount: Optional[float] = None
    budget_currency: Optional[str] = None  # Default to "USD" if not specified
    notes: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if all key fields are collected."""
        required = [
            self.destination,
            self.travel_date,
            self.activities if len(self.activities) > 0 else None,
            self.preferred_brand,
            self.clothes,
            self.budget_amount,
            self.budget_currency,
        ]
        return all(x is not None and (x != "" if isinstance(x, str) else True) for x in required)


def render_intent_md(intent: TravelIntent) -> str:
    """Render the current intent in Markdown for the side panel."""
    activities_str = ", ".join(intent.activities) if intent.activities else "_(not set)_"
    return f"""
### Current Travel Plan (Live Summary)
- **Destination:** {intent.destination or "_(not set)_"}
- **Travel Date:** {intent.travel_date or "_(not set)_"}
- **Activities:** {activities_str}
- **Preferred Brand:** {intent.preferred_brand or "_(not set)_"}
- **Clothes:** {intent.clothes or "_(not set)_"}
- **Budget:** {f"{intent.budget_amount} {intent.budget_currency}" if intent.budget_amount and intent.budget_currency else "_(not set)_"}
- **Notes:** {intent.notes or "_(none)_"}
"""


# ------------------------
# LLM Clarifier Agent
# ------------------------
from datetime import date
CURRENT_DATE = date.today().isoformat()
print("Current date for context:", CURRENT_DATE)

SYSTEM_PROMPT_NEW = f"""
You are a Clarifier Agent for travel planning. Your task:
- Extract or confirm the user's intent across these fields:
  destination, travel_date, activities, preferred_brand, clothes, budget_amount, budget_currency, notes.
- Ask ONLY ONE targeted clarifying question per turn if information is missing or ambiguous.
- If everything is collected and clear, provide a friendly, concise confirmation.
- Use US English tone, be helpful and polite.
- Infer budget currency from context; default to USD.
- If budget is provided without currency, set currency to USD by default.
- Activities should be either an array or strings. If the user gives a single activity (e.g., "hiking"),accept it as a valid array with one item and do not ask for more unless the user seems unsure
- Clothes can be a simple descriptive string (e.g., "casual summer wear").
- Do NOT ask multiple questions at once; keep it single-question per turn.
- Respond concisely and avoid over-prompting.

ontext:
- Today's date (ISO): {CURRENT_DATE}

Rules:
- Keep assistant_message friendly and purposeful.
- Use next_question only if more info is needed AND ask exactly one question.
- Never include extra keys or text outside the JSON.
- please don't consider model date as current date.always use the provided CURRENT_DATE.
- If only one date is mentioned, set start_date = end_date.
- Parse various date formats:   
- Parse ranges like "from 5 March to 8 March", "10-12 Jan 2025", "2025-01-10 to 2025-01-12".
- Support relative: "today", "tomorrow", "this weekend", "next weekend".
- "today" => {CURRENT_DATE}
- "tomorrow" => {CURRENT_DATE} + 1 day
- "this weekend" => Saturday-Sunday of the current week (based on {CURRENT_DATE})
- "next weekend" => Saturday-Sunday of the following week (based on {CURRENT_DATE})
- If month/day is given without year, infer the year with a future bias relative to {CURRENT_DATE}.
- If multiple destinations are mentioned, choose the primary after prepositions (to/in/at/for) or the final city in "heading to …".
- Date format: "YYYY-MM-DD" (single date) or "YYYY-MM-DD to YYYY-MM-DD" (range).

OUTPUT STRICTLY AS A JSON OBJECT with this shape:
{{
  "assistant_message": "string - what the assistant says to the user in this turn",
  "updated_intent": {{
      "destination": "string|null",
      "travel_date": "string|null",
      "activities": ["string", ...]|null,
      "preferred_brand": "string|null",
      "clothes": "string|null",
      "budget_amount": number|null,
      "budget_currency": "string|null",
      "notes": "string|null"
  }},
  "next_question": "string|null"  // null if all fields are complete
}}
"""

def call_llm_clarifier(user_text: str, current_intent: TravelIntent) -> Dict[str, Any]:
    """
    Call Azure OpenAI to clarify and update the intent with a strict JSON response.
    """
    # Convert current intent to a dict ready for the prompt
    intent_dict = asdict(current_intent)
    # Ensure the list exists even if empty
    if intent_dict.get("activities") is None:
        intent_dict["activities"] = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NEW},
        {
            "role": "user",
            "content": json.dumps({
                "user_message": user_text,
                "current_intent": intent_dict
            }, ensure_ascii=False)
        }
    ]

    # Request a JSON object response for reliability
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    # Defensive parsing
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback minimal structure to prevent crashes
        data = {
            "assistant_message": "I'm processing that; could you please clarify one detail?",
            "updated_intent": intent_dict,
            "next_question": "Could you share your destination city or country?"
        }
    return data


def merge_intent(original: TravelIntent, updates: Dict[str, Any]) -> TravelIntent:
    """
    Merge updated_intent dict into the dataclass, performing light normalization.
    """
    merged = TravelIntent(**asdict(original))

    def clean_str(s):
        if s is None:
            return None
        s = str(s).strip()
        return s if s else None

    merged.destination = clean_str(updates.get("destination", merged.destination))
    merged.travel_date = clean_str(updates.get("travel_date", merged.travel_date))

    # Activities: normalize list of strings
    act = updates.get("activities")
    if isinstance(act, list):
        merged.activities = [str(a).strip() for a in act if str(a).strip()]
    elif isinstance(act, str):
        # Split by comma if a single string is provided
        merged.activities = [a.strip() for a in act.split(",") if a.strip()]

    merged.preferred_brand = clean_str(updates.get("preferred_brand", merged.preferred_brand))
    merged.clothes = clean_str(updates.get("clothes", merged.clothes))

    # Budget
    budget_amount = updates.get("budget_amount", merged.budget_amount)
    try:
        merged.budget_amount = float(budget_amount) if budget_amount is not None else merged.budget_amount
    except (ValueError, TypeError):
        # if parsing fails, keep original
        pass

    budget_currency = updates.get("budget_currency", merged.budget_currency)
    if budget_currency is None and merged.budget_amount is not None and merged.budget_currency is None:
        # Default to USD if amount is present but currency missing
        merged.budget_currency = "USD"
    else:
        merged.budget_currency = clean_str(budget_currency) or merged.budget_currency

    merged.notes = clean_str(updates.get("notes", merged.notes))

    # Light date validation (optional): allows single date or date range
    if merged.travel_date:
        # Accept YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD
        pattern_single = r"^\d{4}-\d{2}-\d{2}$"
        pattern_range = r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$"
        if not (re.match(pattern_single, merged.travel_date) or re.match(pattern_range, merged.travel_date)):
            # Leave as-is (LLM might have better text), or add a hint in notes
            merged.notes = (merged.notes or "")
            if "format" not in (merged.notes or "").lower():
                merged.notes += " (Travel date format pending confirmation)"

    return merged


# ------------------------
# Gradio App Logic
# ------------------------
INTRO_MD = """
## ✈️ Travel Intent Clarifier
Tell me about your trip, and I'll help capture the essentials: **destination, travel dates, activities, preferred brand, clothes, and budget**.

I'll ask **one concise follow-up** at a time until we have a complete plan.  
You can also paste existing details; I'll extract and organize them.
"""

def bot_turn(user_message: str, history: List[List[str]], state: Dict[str, Any]):
    """
    Handle a single user message:
    - Call the clarifier agent.
    - Update intent state.
    - Produce assistant reply.
    - Return updated chat history and summary markdown.
    """
    if "intent" not in state or state["intent"] is None:
        state["intent"] = TravelIntent()

    # Call LLM
    llm_result = call_llm_clarifier(user_message, state["intent"])
    assistant_message = llm_result.get("assistant_message") or "Thanks! Could you share more details?"
    updates = llm_result.get("updated_intent") or {}
    next_question = llm_result.get("next_question")

    # Merge intent updates
    state["intent"] = merge_intent(state["intent"], updates)

    # Compose assistant output:
    if next_question:
        assistant_text = f"{assistant_message}\n\n**Question:** {next_question}"
    else:
        # Completed or no pending question
        if state["intent"].is_complete():
            assistant_text = (
                f"{assistant_message}\n\n"
                f"Looks like we have everything needed. 🎉\n"
                f"If you want to refine anything (e.g., change brand or add more activities), just say so."
            )
        else:
            assistant_text = assistant_message

    # Update chat history
    
    history = history + [
           {"role": "user", "content": user_message},
           {"role": "assistant", "content": assistant_text},
       ]


    # Update summary panel
    summary_md = render_intent_md(state["intent"])

    return history, summary_md, state, gr.update(value="")


def clear_chat():
    return [], render_intent_md(TravelIntent()), {"intent": TravelIntent()}


with gr.Blocks(title="Travel Intent Clarifier") as demo:
    gr.Markdown(INTRO_MD)
    with gr.Row():
        chatbot = gr.Chatbot(
            label="Chat",
            type="messages",
            height=480,
            avatar_images=("🧑‍💼", "🤖"),
            allow_tags=False,  # ✅ Explicitly disable tags for Gradio 6.0+
        )
        summary = gr.Markdown(render_intent_md(TravelIntent()), elem_id="summary_panel")

    with gr.Row():
        msg = gr.Textbox(placeholder="Hi! I want to go to Miami in March...", scale=8)
        send = gr.Button("Send", variant="primary", scale=1)
        reset = gr.Button("Clear", variant="secondary", scale=1)

    state = gr.State({"intent": TravelIntent()})

    # Wire events
    send.click(
        fn=bot_turn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, summary, state, msg],
    )
    msg.submit(
        fn=bot_turn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, summary, state, msg],
    )
    reset.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, summary, state, msg],
    )

if __name__ == "__main__":
    # For local dev; set share=True if you need a public link
    demo.launch()

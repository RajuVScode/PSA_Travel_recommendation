
import os
import json
from datetime import datetime, timedelta
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


from datetime import datetime

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

system_prompt = f"""
You extract travel intent from user text and return JSON ONLY with:
- destination: string or null
- start_date: YYYY-MM-DD or null
- end_date: YYYY-MM-DD or null
- duration_days: integer or null
- season_hint: string or null (e.g., "summer", "winter", "spring", "fall")  

Context:
- Today's date (ISO): {CURRENT_DATE}

Rules:
- If only one date is mentioned, set start_date = end_date.
- duration_days: calculate based on start_date and end_date.
- season_hint: infer from month if possible (e.g., Dec-Feb => "winter").
- Parse various date formats:   
- Parse ranges like "from 5 March to 8 March", "10–12 Jan 2025", "2025-01-10 to 2025-01-12".
- Support relative: "today", "tomorrow", "this weekend", "next weekend".
  - "today" => {CURRENT_DATE}
  - "tomorrow" => {CURRENT_DATE} + 1 day
  - "this weekend" => Saturday–Sunday of the current week (based on {CURRENT_DATE})
  - "next weekend" => Saturday–Sunday of the following week (based on {CURRENT_DATE})
- If month/day is given without year, infer the year with a future bias relative to {CURRENT_DATE}.
- If multiple destinations are mentioned, choose the primary after prepositions (to/in/at/for) or the final city in "heading to …".
- Output must be valid JSON only. No comments, no trailing commas, no extra fields.

Examples:
User: "Planning a trip to Goa from Jan 10 to Jan 12, 2025"
JSON: {{"destination":"Goa","start_date":"2025-01-10","end_date":"2025-01-12"}}

User: "I'll be in Paris 5–8 March"
JSON: {{"destination":"Paris","start_date":"<infer-future-YYYY>-03-05","end_date":"<infer-future-YYYY>-03-08"}}

User: "Client visit to Bangalore tomorrow"
JSON: {{"destination":"Bangalore","start_date":"<compute-from {CURRENT_DATE}>","end_date":"<compute-from {CURRENT_DATE}>"}}
"""


def _resolve_relative_dates(obj, today=None):
    # Replace placeholders <TODAY>, <TOMORROW>, <THIS_WEEKEND>, <NEXT_WEEKEND>
    if today is None:
        today = datetime.now()
    def weekend_range(base_date):
        sat = base_date + timedelta(days=(5 - base_date.weekday()) % 7)
        sun = sat + timedelta(days=1)
        return sat.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")

    sd, ed = obj.get("start_date"), obj.get("end_date")
    if sd == "<TODAY>" or ed == "<TODAY>":
        d = today.strftime("%Y-%m-%d")
        obj["start_date"] = d; obj["end_date"] = d
    if sd == "<TOMORROW>" or ed == "<TOMORROW>":
        d = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        obj["start_date"] = d; obj["end_date"] = d
    if sd == "<THIS_WEEKEND>" or ed == "<THIS_WEEKEND>":
        s,e = weekend_range(today)
        obj["start_date"] = s; obj["end_date"] = e
    if sd == "<NEXT_WEEKEND>" or ed == "<NEXT_WEEKEND>":
        s,e = weekend_range(today + timedelta(days=7))
        obj["start_date"] = s; obj["end_date"] = e
    return obj

def extract_with_llm(user_prompt: str) -> dict:
        
    """
    Extract destination, start date, end date, duration, and season hint from user prompt.
    """
    destination = None
    start_date = None
    end_date = None
    duration_days = None
    season_hint = None

    try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Prompt: {system_prompt}"},
                    {"role": "user", "content": f"Prompt: {user_prompt}"}
                ],
                temperature=0.0,
                max_tokens=50,
            )
            result = json.loads(response.choices[0].message.content)
            destination = result.get("destination")
            start_date = result.get("start_date")
            end_date = result.get("end_date")
            season_hint = result.get("season_hint")
            duration_days = result.get("duration_days")
    except Exception as e:
            print(f"[DEBUG] LLM destination extraction failed: {e}")
            destination = None

    print(f"\n[DEBUG parseTravelIntent]")
    print(f"  Destination: {destination}")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Duration Days: {duration_days}")
    print(f"  Season Hint: {season_hint}")
    
    result = _resolve_relative_dates(result)
    return result

if __name__ == "__main__":
    user_input = input("Enter your travel request: ")

    plan = extract_with_llm(user_input)
    print(plan)
    
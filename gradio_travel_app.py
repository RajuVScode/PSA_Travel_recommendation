"""
Gradio UI for Travel Recommendation Pipeline
Provides a user-friendly interface for generating formal travel plans.
"""

import gradio as gr
from gradio_calendar import Calendar
from main import generate_travel_recommendation


def generate_plan(user_prompt, destination_override=None, start_date=None, end_date=None):
    """
    Wrapper function that calls the travel recommendation pipeline.
    Handles optional overrides for destination and dates. Date inputs may be
    `datetime.date` objects or ISO strings; we normalize them to `YYYY-MM-DD`.
    """
    try:
        # Normalize dates to YYYY-MM-DD when provided
        def _fmt_date(d):
            if d is None or d == "":
                return None
            # If Gradio returns a date object
            try:
                from datetime import date, datetime
                if isinstance(d, date) and not isinstance(d, datetime):
                    return d.isoformat()
            except Exception:
                pass
            # If it's already a string, assume it's ISO-like and return trimmed
            if isinstance(d, str):
                return d.strip()
            # Fallback: try to convert using fromisoformat
            try:
                return datetime.fromisoformat(str(d)).date().isoformat()
            except Exception:
                return str(d)

        sd = _fmt_date(start_date)
        ed = _fmt_date(end_date)

        # Append explicit dates/destination to the prompt so parser will pick them up
        augmented_prompt = (user_prompt or "").strip()
        if destination_override:
            augmented_prompt = f"{augmented_prompt} Destination: {destination_override}"
        if sd:
            augmented_prompt = f"{augmented_prompt} Start Date: {sd}"
        if ed:
            augmented_prompt = f"{augmented_prompt} End Date: {ed}"

        result = generate_travel_recommendation(augmented_prompt)
        return result, ""
    except Exception as e:
        error_msg = f"Error generating plan: {str(e)}"
        return "", error_msg


def clear_outputs():
    """Clear output fields."""
    return "", ""


# Build the Gradio interface
with gr.Blocks(title="Travel Recommendation Planner") as demo:
    gr.Markdown("""
    # 🌍 Travel Recommendation Planner
    
    Get a personalized, formal travel plan with weather insights and product recommendations.
    
    Enter your travel details below and we'll generate a comprehensive itinerary tailored to your destination and dates.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📋 Describe your travel")
            
            travel_prompt = gr.Textbox(
                label="Travel Request",
                placeholder="e.g., 'I'm travelling to the UK for 2 weeks in April'",
                lines=5,
                info="Describe your travel plans, including destination, dates, and any preferences."
            )
            
            with gr.Column():
                destination_input = gr.Textbox(
                    label="Destination (Optional Override)",
                    placeholder="e.g., 'London'",
                    interactive=True,
                    info=""
                )
                start_date_input = Calendar(
                    label="Start Date (Optional)",
                    info=".",
                    value=None                    
                )
                end_date_input = Calendar(  
                    label="End Date (Optional)",
                    info="",
                    value=None
                )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate Plan", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("## ✅ Recommended Travel Plan")
            
            plan_output = gr.Textbox(
                label="Travel Plan",
                lines=15,
                interactive=False,
            )
            
            error_output = gr.Textbox(
                label="Errors (if any)",
                lines=3,
                interactive=False,
                visible=True,
            )
    
    gr.Markdown("""
    ---
    ### ℹ️ How to use:
    1. **Describe your travel**: Include destination, duration, dates, and any preferences
    2. **Optional overrides**: Manually specify destination or dates if needed
    3. **Generate**: Click the button to create your personalized plan
    
    The system will:
    - Extract destination, dates, and duration from your description
    - Fetch real-time or seasonal weather data
    - Retrieve top 5 relevant products from the catalog
    - Generate a formal itinerary with weather insights, activities, and product recommendations
    """)
    
    # Event handlers
    submit_btn.click(
        fn=generate_plan,
        inputs=[travel_prompt, destination_input, start_date_input, end_date_input],
        outputs=[plan_output, error_output]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        outputs=[plan_output, error_output]
    )
    
    # Auto-focus on travel prompt when page loads
    travel_prompt.focus()


if __name__ == "__main__":
    demo.launch()

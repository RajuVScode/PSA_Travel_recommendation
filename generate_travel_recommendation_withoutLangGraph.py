#checking again
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
    
   # from outfit_planner import recommend_outfits
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

    outfit_recommendations = recommend_outfits_llm(
        activities=activities_list,
        duration_days=parsed_intent.get("duration_days"),
        intention="balanced",
        climate=None,
        laundry_available=False,
        weather=weather,
        destination=parsed_intent.get("destination"),
        start_date=parsed_intent.get("start_date"),
        end_date=parsed_intent.get("end_date"),
        extra_context=events
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

if __name__ == "__main__":
    user_input = input("Enter your travel request: ")

    plan = generate_travel_recommendation(user_input)    
    print("\n" + plan)
    print("\n" + "=" * 80)


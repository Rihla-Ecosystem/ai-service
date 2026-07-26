import json
import structlog
from typing import Any, Dict, List, Optional

from app.config import settings

logger = structlog.get_logger()

EGYPT_CITIES = {
    "cairo": {"lat": 30.0444, "lon": 31.2357},
    "giza": {"lat": 29.9870, "lon": 31.2118},
    "alexandria": {"lat": 31.2001, "lon": 29.9187},
    "luxor": {"lat": 25.6872, "lon": 32.6396},
    "aswan": {"lat": 24.0889, "lon": 32.8998},
    "hurghada": {"lat": 27.2579, "lon": 33.8116},
    "sharm el sheikh": {"lat": 27.9158, "lon": 34.3299},
    "dahab": {"lat": 28.5095, "lon": 34.5165},
    "siwa": {"lat": 29.2032, "lon": 25.5195},
    "suez": {"lat": 29.9668, "lon": 32.5498},
    "port said": {"lat": 31.2653, "lon": 32.3019},
    "ismailia": {"lat": 30.6043, "lon": 32.2722},
    "tanta": {"lat": 30.7865, "lon": 31.0004},
    "mansoura": {"lat": 31.0409, "lon": 31.3785},
    "fayoum": {"lat": 29.3571, "lon": 30.8410},
    "minya": {"lat": 28.1099, "lon": 30.7503},
    "asjut": {"lat": 27.1809, "lon": 31.1837},
    "sohag": {"lat": 26.5569, "lon": 31.6948},
    "qena": {"lat": 26.1642, "lon": 32.7267},
    "abu simbel": {"lat": 22.3457, "lon": 31.6165},
}

TOOL_DEFINITIONS = [
    {
        "name": "get_nearby_attractions",
        "description": "Find tourist attractions near a given latitude and longitude within a radius in meters.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "radius": {"type": "integer", "description": "Search radius in meters (default 1000)"},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "search_attractions",
        "description": "Search for attractions by keyword, category, or city name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term (e.g. 'mosque', 'museum', 'pyramids')"},
                "category": {"type": "string", "description": "Optional category filter (mosque, museum, heritage_site, etc.)"},
                "city": {"type": "string", "description": "Optional city name filter (Cairo, Luxor, Aswan, etc.)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_safety_info",
        "description": "Get current safety and risk information for an Egyptian city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name (Cairo, Giza, Alexandria, Luxor, Aswan, Hurghada, etc.)"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_emergency_contacts",
        "description": "Get emergency contact numbers and procedures for a given situation.",
        "parameters": {
            "type": "object",
            "properties": {
                "context_type": {
                    "type": "string",
                    "enum": ["medical", "police", "harassment", "fire", "general"],
                    "description": "Type of emergency situation",
                },
            },
            "required": ["context_type"],
        },
    },
    {
        "name": "get_legal_guidelines",
        "description": "Get Egyptian laws and regulations for tourists on a specific topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["photography", "alcohol", "dress_code", "antiquities", "public_morality", "citizenship"],
                    "description": "Legal topic to look up",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_currency_info",
        "description": "Get information about Egyptian currency and current exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "denomination": {
                    "type": "string",
                    "description": "Optional: specific banknote or coin (e.g. '10 EGP', '50 EGP', '1 pound coin')",
                },
                "base_currency": {
                    "type": "string",
                    "description": "Optional: your home currency code for exchange rate (e.g. 'USD', 'EUR', 'GBP')",
                },
            },
        },
    },
    {
        "name": "get_scam_warnings",
        "description": "Get common scam warnings and countermeasures for a category or specific situation.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["transportation", "airport", "sites", "shopping", "dining", "social"],
                    "description": "Category of scams to look up",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Optional: filter by severity level",
                },
            },
        },
    },
    {
        "name": "recommend_itinerary",
        "description": "Create a complete day-by-day travel itinerary for Egypt. Suggests cities based on interests, fetches attractions, safety data, and composes a structured plan with budget estimates, timing, and safety tips.",
        "parameters": {
            "type": "object",
            "properties": {
                "interests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "User interests e.g. ['history', 'photography', 'food', 'beach', 'shopping', 'adventure']",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for the trip (1-14)",
                },
                "budget": {
                    "type": "string",
                    "enum": ["budget", "mid", "luxury"],
                    "description": "Budget level for the trip",
                },
                "cities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific cities to include. If empty, AI will suggest based on interests.",
                },
                "style": {
                    "type": "string",
                    "enum": ["cultural", "adventure", "relaxation", "family", "solo", "romantic"],
                    "description": "Optional: travel style",
                },
                "base_currency": {
                    "type": "string",
                    "description": "Optional: user's home currency code for budget estimates (e.g. 'USD', 'EUR', 'GBP')",
                },
            },
            "required": ["interests", "days", "budget"],
        },
    },
]


async def call_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    logger.info("Tool call", tool=tool_name, args=arguments)

    if tool_name == "get_nearby_attractions":
        return await _get_nearby_attractions(**arguments)
    elif tool_name == "search_attractions":
        return await _search_attractions(**arguments)
    elif tool_name == "get_safety_info":
        return await _get_safety_info(**arguments)
    elif tool_name == "get_emergency_contacts":
        return await _get_emergency_contacts(**arguments)
    elif tool_name == "get_legal_guidelines":
        return await _get_legal_guidelines(**arguments)
    elif tool_name == "get_currency_info":
        return await _get_currency_info(**arguments)
    elif tool_name == "get_scam_warnings":
        return await _get_scam_warnings(**arguments)
    elif tool_name == "recommend_itinerary":
        return await _recommend_itinerary(**arguments)

    return f"Unknown tool: {tool_name}"


async def _get_nearby_attractions(lat: float, lon: float, radius: int = 1000) -> str:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.gis_service_url}/api/v1/nearby-sites",
                params={"lat": lat, "lon": lon, "radius": radius},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    sites = []
                    for s in data[:10]:
                        name = s.get("name", "Unknown")
                        cat = ", ".join(s.get("categories", []))
                        sites.append(f"- {name} ({cat})")
                    return "Nearby attractions:\n" + "\n".join(sites)
                return "No nearby attractions found."
            return f"Error fetching attractions: {resp.status_code}"
    except Exception as e:
        return f"Unable to fetch nearby attractions: {str(e)}"


async def _search_attractions(query: str, category: str = "", city: str = "") -> str:
    from app.main import vector_store
    from app.rag.retriever import retrieve

    if not vector_store:
        return "Vector store not available."

    results = await retrieve(vector_store, query, "attractions", top_k=5)
    if not results:
        return f"No attractions found for '{query}'."

    lines = [f"Top results for '{query}':"]
    for r in results:
        text = r.get("text", "")[:200]
        score = r.get("score", 0)
        lines.append(f"- [score:{score:.2f}] {text}")
    return "\n".join(lines)


async def _get_safety_info(city: str) -> str:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.risk_service_url}/safety/current",
                params={"city": city.lower()},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return f"Safety info for {city}:\n{data}"
            return f"No safety data for {city}."
    except Exception as e:
        return f"Unable to fetch safety info: {str(e)}"


async def _get_emergency_contacts(context_type: str) -> str:
    from app.main import vector_store
    from app.rag.retriever import retrieve

    if not vector_store:
        return "Vector store not available."

    results = await retrieve(vector_store, f"{context_type} emergency contacts", "emergency", top_k=3)
    if not results:
        return f"No emergency contacts found for '{context_type}'."

    lines = [f"Emergency contacts for {context_type}:"]
    for r in results:
        lines.append(r.get("text", ""))
    return "\n".join(lines)


async def _get_legal_guidelines(topic: str) -> str:
    from app.main import vector_store
    from app.rag.retriever import retrieve

    if not vector_store:
        return "Vector store not available."

    topic_map = {
        "photography": "photography drone law",
        "alcohol": "alcohol substance law",
        "dress_code": "dress code public decorum",
        "antiquities": "antiquities protection heritage",
        "public_morality": "public morality social framework",
        "citizenship": "citizenship nationality law",
    }
    query = topic_map.get(topic, topic)
    results = await retrieve(vector_store, query, "legal", top_k=3)
    if not results:
        return f"No legal guidelines found for '{topic}'."

    lines = [f"Legal guidelines for {topic} in Egypt:"]
    for r in results:
        lines.append(r.get("text", ""))
    return "\n".join(lines)


async def _get_currency_info(denomination: str = "", base_currency: str = "") -> str:
    from app.main import vector_store
    from app.rag.retriever import retrieve

    if not vector_store:
        return "Vector store not available."

    query = f"Egyptian currency {denomination}" if denomination else "Egyptian pound currency info"
    results = await retrieve(vector_store, query, "currency", top_k=5)
    lines = []

    if results:
        lines.append("Egyptian Currency Information:")
        for r in results:
            lines.append(r.get("text", ""))

    if base_currency:
        lines.append(f"\nNote: Live exchange rate for {base_currency} to EGP would be fetched from exchange rate API.")

    return "\n".join(lines) if lines else "Currency information not available."


async def _get_scam_warnings(category: str = "", severity: str = "") -> str:
    from app.main import vector_store
    from app.rag.retriever import retrieve

    if not vector_store:
        return "Vector store not available."

    query = f"{category} scam {severity}" if category else "scam warnings Egypt"
    results = await retrieve(vector_store, query, "scams", top_k=5)
    if not results:
        return "No scam warnings found."

    lines = [f"Scam warnings for '{category}':"]
    for r in results:
        lines.append(r.get("text", ""))
    return "\n".join(lines)


async def _suggest_cities(interests: list[str], days: int, style: str) -> list[str]:
    from app.main import llm_client

    prompt = (
        "You are an Egypt travel expert. Based on the following traveler profile, "
        "suggest the best Egyptian cities to visit. Return ONLY a JSON array of city names, "
        "e.g. [\"Cairo\", \"Luxor\"].\n\n"
        f"Interests: {', '.join(interests)}\n"
        f"Days available: {days}\n"
        f"Travel style: {style}\n\n"
        "Rules:\n"
        "- Return 2-4 cities max\n"
        "- Cities must be real Egyptian cities\n"
        "- Order by logical travel flow (e.g. Cairo → Luxor → Aswan)\n"
        "- Only valid cities: Cairo, Giza, Alexandria, Luxor, Aswan, Hurghada, "
        "Sharm el Sheikh, Dahab, Siwa, Abu Simbel, Fayoum\n"
        "- Return ONLY the JSON array, no other text"
    )

    resp = await llm_client.generate(system_prompt="You are a helpful Egypt travel expert.", user_message=prompt)
    text = ""
    if resp is not None and hasattr(resp, "text") and resp.text is not None:
        text = resp.text.strip()
    if not text:
        return ["cairo", "luxor"]
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        cities = json.loads(text)
        if isinstance(cities, list) and len(cities) > 0:
            return [c.lower() for c in cities]
    except (json.JSONDecodeError, TypeError):
        pass

    return ["cairo", "luxor"]


async def _recommend_itinerary(
    interests: list[str],
    days: int,
    budget: str,
    cities: Optional[list[str]] = None,
    style: str = "cultural",
    base_currency: str = "",
) -> str:
    from app.main import llm_client

    if cities is None or len(cities) == 0:
        suggested = await _suggest_cities(interests, days, style)
        cities = suggested
        logger.info("Cities suggested by AI", cities=cities)

    city_data = {}
    import asyncio

    async def fetch_city_data(city: str) -> tuple[str, dict]:
        city_lower = city.lower().strip()
        coords = EGYPT_CITIES.get(city_lower, EGYPT_CITIES.get("cairo"))

        tasks = {}

        attractions_query = " ".join(interests) + f" attractions in {city}"
        tasks["attractions"] = _search_attractions(query=attractions_query, city=city)

        tasks["safety"] = _get_safety_info(city=city_lower)

        interest_categories = set()
        interest_to_scam = {
            "history": "sites",
            "photography": "sites",
            "shopping": "shopping",
            "food": "dining",
            "nightlife": "social",
        }
        for interest in interests:
            cat = interest_to_scam.get(interest.lower())
            if cat:
                interest_categories.add(cat)
        for cat in interest_categories:
            tasks[f"scam_{cat}"] = _get_scam_warnings(category=cat)

        if coords:
            tasks["nearby"] = _get_nearby_attractions(
                lat=coords["lat"],
                lon=coords["lon"],
                radius=5000,
            )

        results = {}
        for key, coro in tasks.items():
            try:
                results[key] = await coro
            except Exception as e:
                results[key] = f"Error: {str(e)}"

        return city, results

    fetch_tasks = [asyncio.create_task(fetch_city_data(c)) for c in cities]
    for coro in asyncio.as_completed(fetch_tasks):
        city, data = await coro
        city_data[city] = data

    currency_info = await _get_currency_info(base_currency=base_currency)

    budget_guides = {
        "budget": "street food, public transport, budget hotels/hostels, free attractions",
        "mid": "mix of street food and restaurants, Uber/taxis, 3-4 star hotels, paid entry sites",
        "luxury": "fine dining, private drivers, 5-star resorts, VIP entry, Nile cruises",
    }
    budget_desc = budget_guides.get(budget, budget_guides["mid"])

    city_sections = []
    for city, data in city_data.items():
        section = f"--- {city.title()} ---\n"
        section += f"Attractions:\n{data.get('attractions', 'No data')}\n"
        section += f"Safety:\n{data.get('safety', 'No data')}\n"
        if data.get("nearby"):
            section += f"Nearby:\n{data['nearby']}\n"
        for key in data:
            if key.startswith("scam_"):
                section += f"{data[key]}\n"
        city_sections.append(section)

    system_prompt = (
        "You are an expert Egypt travel itinerary planner. Create a detailed, day-by-day itinerary. "
        "You must follow these rules:\n"
        "1. NEVER mention or suggest visiting military sites, restricted areas, or border zones\n"
        "2. NEVER disparage Egypt, its government, or its people\n"
        "3. Frame every safety warning positively: 'Stay safe by doing X' not 'Avoid Y'\n"
        "4. Every safety/scam warning MUST be paired with a concrete countermeasure\n"
        "5. Be enthusiastic and highlight Egypt's rich culture, history, and hospitality\n"
        "6. Include practical tips: dress codes, bargaining culture, photography etiquette\n"
        "7. Provide budget estimates in EGP and the user's currency if known\n"
        "8. Mention best times of day for each activity (e.g. mornings for outdoor sites)\n"
        "9. Include meal suggestions and restaurant recommendations where appropriate\n"
        "10. Keep a realistic pace — don't cram too much in one day"
    )

    compose_prompt = (
        f"Create a {days}-day itinerary for a trip to {', '.join(c.title() for c in cities)}.\n\n"
        f"Traveler profile:\n"
        f"- Interests: {', '.join(interests)}\n"
        f"- Budget: {budget} ({budget_desc})\n"
        f"- Style: {style}\n"
        f"- Base currency: {base_currency or 'EGP'}\n\n"
        f"City data collected:\n"
        + "\n".join(city_sections)
        + f"\n\nCurrency information:\n{currency_info}\n\n"
        + "Return your response in the following JSON structure with NO additional text:\n"
        + json.dumps({
            "markdown": "Full markdown itinerary with # headings, **bold**, - lists, and detailed day-by-day plan",
            "json": {
                "title": "Trip title",
                "budget_estimate": {"egp": 0, "usd": 0},
                "currency_note": "Exchange rate note",
                "days": [
                    {
                        "day": 1,
                        "city": "City name",
                        "theme": "Day theme",
                        "items": [
                            {
                                "time": "HH:MM",
                                "activity": "Activity name",
                                "type": "attraction|meal|transport|rest|other",
                                "fee_egp": 0,
                                "duration_hours": 1.0,
                                "safety_tip": "Safety tip",
                                "scam_warning": "Scam warning if any",
                            }
                        ],
                    }
                ],
                "trip_notes": ["Note 1", "Note 2"],
            }
        }, indent=2)
        + "\n\nIMPORTANT: Return ONLY valid JSON matching the structure above. No other text."
    )

    resp = await llm_client.generate(system_prompt=system_prompt, user_message=compose_prompt)
    text = ""
    if resp is not None and hasattr(resp, "text") and resp.text is not None:
        text = resp.text.strip()
    if not text:
        return "Unable to generate itinerary. Please try again."
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(text)
        markdown = result.get("markdown", text)
        structured = result.get("json", result)

        budget_note = ""
        if base_currency:
            budget_note = (
                f"\n\n> 💰 Budget estimates shown in EGP. "
                f"Exchange rate note: {structured.get('currency_note', 'Rates fluctuate daily.')}"
            )

        return (
            markdown
            + budget_note
            + "\n\n<!-- structured: "
            + json.dumps(structured, ensure_ascii=False)
            + " -->"
        )
    except (json.JSONDecodeError, TypeError):
        return text

from app.core.system_prompt import build_system_prompt
from app.agent.tools import TOOL_DEFINITIONS, call_tool


PERSONA_NAME = "tour_guide"
ALLOWED_TOOLS = ["search_attractions", "get_nearby_attractions", "get_currency_info", "get_legal_guidelines", "recommend_itinerary"]


async def handle(message: str, context: dict | None = None) -> str:
    from app.main import llm_client

    system_prompt = build_system_prompt(persona=PERSONA_NAME, context=context)
    tools = [t for t in TOOL_DEFINITIONS if t["name"] in ALLOWED_TOOLS]

    response = await llm_client.generate_with_tools(
        system_prompt=system_prompt,
        user_message=message,
        tools=tools,
    )

    if hasattr(response, "function_calls") and response.function_calls:
        tool_results = []
        for fc in response.function_calls:
            result = await call_tool(fc.name, dict(fc.args))
            tool_results.append(result)

        combined = message + "\n\nTool results:\n" + "\n".join(tool_results)
        response = await llm_client.generate(
            system_prompt=system_prompt,
            user_message=combined,
        )

    if response is not None and hasattr(response, "text") and response.text is not None:
        return response.text
    return str(response) if response is not None else ""

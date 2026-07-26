import structlog
from typing import Any, Dict, Optional

from app.core.system_prompt import build_system_prompt
from app.core.guardrails import check_input, check_output, sanitize_output
from app.agent.tools import TOOL_DEFINITIONS, call_tool

logger = structlog.get_logger()

INTENT_KEYWORDS = {
    "tour_guide": ["history", "attraction", "site", "museum", "pyramid", "temple", "monument",
                   "tour", "visit", "see", "ancient", "pharaoh", "culture", "heritage"],
    "safety_guru": ["safe", "danger", "scam", "risk", "emergency", "warning", "police",
                    "hospital", "ambulance", "crime", "protect", "avoid", "alert"],
    "local_expert": ["food", "eat", "restaurant", "shop", "bargain", "haggle", "local",
                     "insider", "tip", "hidden", "best", "authentic", "custom", "culture"],
}


def detect_intent(message: str) -> str:
    msg_lower = message.lower()
    scores = {"tour_guide": 0, "safety_guru": 0, "local_expert": 0}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                scores[intent] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "tour_guide"

    winners = [k for k, v in scores.items() if v == max_score]
    return winners[0]


def _safe_text(response) -> str:
    if response is None:
        return ""
    if hasattr(response, "text") and response.text is not None:
        return response.text
    return str(response)


async def route_and_respond(
    message: str,
    persona: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    guard_result = check_input(message)
    if guard_result.blocked:
        logger.warning("Input blocked by guardrails", reason=guard_result.reason)
        return {
            "response": (
                "I'm here to help with tourism information about Egypt. "
                "Let's keep our conversation focused on making your visit "
                "to Egypt wonderful and safe! How can I assist you today?"
            ),
            "blocked": True,
            "reason": guard_result.reason,
        }

    if persona == "auto":
        persona = detect_intent(message)
        logger.info("Auto-detected persona", persona=persona)

    system_prompt = build_system_prompt(persona=persona, context=context)

    from app.main import llm_client

    if not llm_client:
        return {"response": "AI service is not fully initialized yet."}

    try:
        response = await llm_client.generate_with_tools(
            system_prompt=system_prompt,
            user_message=message,
            tools=TOOL_DEFINITIONS,
        )

        if response is None:
            return {"response": "I couldn't generate a response. Please try again."}

        if hasattr(response, "function_calls") and response.function_calls:
            tool_results = []
            for fc in response.function_calls:
                result = await call_tool(fc.name, dict(fc.args))
                tool_results.append(result)

            combined_message = message + "\n\nTool results:\n" + "\n".join(tool_results)
            response = await llm_client.generate(
                system_prompt=system_prompt,
                user_message=combined_message,
            )

        text = _safe_text(response)

        guard_result = check_output(text)
        if guard_result.requires_regeneration:
            logger.warning("Output required regeneration", reason=guard_result.reason)
            response = await llm_client.generate(
                system_prompt=system_prompt + "\n\nIMPORTANT: Do not mention restricted areas.",
                user_message=message,
            )
            text = _safe_text(response)

        if guard_result.modified:
            text = sanitize_output(text)

        return {"response": text, "persona": persona}

    except Exception as e:
        logger.error("Agent processing failed", error=str(e))
        return {"response": "I encountered an issue processing your request. Please try again."}

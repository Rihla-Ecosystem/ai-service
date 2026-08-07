import pytest
from app.agent.tools import TOOL_DEFINITIONS, EGYPT_CITIES


class TestTools:
    def test_tool_definitions_exist(self):
        assert len(TOOL_DEFINITIONS) >= 8

    def test_each_tool_has_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "type" in tool["parameters"]
            assert "properties" in tool["parameters"]

    def test_get_nearby_attractions_params(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_nearby_attractions")
        props = tool["parameters"]["properties"]
        assert "lat" in props
        assert "lon" in props

    def test_search_attractions_params(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_attractions")
        props = tool["parameters"]["properties"]
        assert "query" in props

    def test_get_safety_info_params(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_safety_info")
        props = tool["parameters"]["properties"]
        assert "city" in props

    def test_get_emergency_contacts_has_enum(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_emergency_contacts")
        props = tool["parameters"]["properties"]
        assert "enum" in props["context_type"]
        assert "medical" in props["context_type"]["enum"]

    def test_get_legal_guidelines_topic_enum(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_legal_guidelines")
        props = tool["parameters"]["properties"]
        assert "enum" in props["topic"]
        assert "photography" in props["topic"]["enum"]

    def test_get_scam_warnings_category_enum(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_scam_warnings")
        props = tool["parameters"]["properties"]
        assert "enum" in props["category"]
        assert "transportation" in props["category"]["enum"]

    def test_get_currency_info_optional_params(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_currency_info")
        props = tool["parameters"]["properties"]
        assert "denomination" in props
        assert "base_currency" in props

    def test_recommend_itinerary_params(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "recommend_itinerary")
        props = tool["parameters"]["properties"]
        assert "interests" in props
        assert "days" in props
        assert "budget" in props
        assert "cities" in props
        assert props["budget"]["enum"] == ["budget", "mid", "luxury"]

    def test_egypt_cities_cairo(self):
        assert EGYPT_CITIES["cairo"]["lat"] == 30.0444
        assert EGYPT_CITIES["cairo"]["lon"] == 31.2357

    def test_egypt_cities_major(self):
        major = {"cairo", "luxor", "aswan", "alexandria", "hurghada", "giza"}
        assert major.issubset(EGYPT_CITIES.keys())

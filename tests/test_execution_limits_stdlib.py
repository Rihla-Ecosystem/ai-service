"""Dependency-free regression tests for operation-scoped execution budgets."""

import unittest

from app.core.execution_limits import (
    AI_CHAT_QUERY,
    AI_EXECUTION_POLICIES,
    AI_IMAGE_ANALYSIS,
    AI_TRIP_ITINERARY,
    CHAT_MAX_INPUT_TOKENS,
    ExecutionLimitExceeded,
    REAL_TIME_TRANSLATION,
    begin_execution_budget,
    current_execution_budget,
    end_execution_budget,
    enforce_input_budget,
    estimate_text_tokens,
    output_limit,
    record_input_tokens,
    record_output_tokens,
)


class ExecutionLimitTests(unittest.TestCase):
    def tearDown(self):
        end_execution_budget()

    def test_feature_policies_are_central_and_installed_at_operation_start(self):
        expected = {
            AI_CHAT_QUERY: (6000, 800),
            AI_IMAGE_ANALYSIS: (3000, 400),
            REAL_TIME_TRANSLATION: (1000, 500),
            AI_TRIP_ITINERARY: (8000, 1000),
        }
        self.assertEqual(
            {feature: (policy["max_input_tokens"], policy["max_output_tokens"])
             for feature, policy in AI_EXECUTION_POLICIES.items()},
            expected,
        )
        for feature, limits in expected.items():
            budget = begin_execution_budget(feature)
            self.assertEqual((budget.max_input_tokens, budget.max_output_tokens), limits)
            end_execution_budget()

    def test_multi_call_budget_uses_actual_output(self):
        begin_execution_budget(AI_CHAT_QUERY)
        self.assertEqual(output_limit(1200), 800)
        record_output_tokens(300, "TEXT_CHAT")
        self.assertEqual(output_limit(1200), 500)
        record_output_tokens(300, "TEXT_CHAT")
        self.assertEqual(output_limit(1200), 200)
        record_output_tokens(200, "TEXT_CHAT")
        with self.assertRaises(ExecutionLimitExceeded):
            output_limit(1200)

    def test_final_provider_visible_input_is_preflight_limited(self):
        begin_execution_budget(AI_CHAT_QUERY)
        system = "s" * 4000
        user = "u" * 20_000
        self.assertLessEqual(estimate_text_tokens(system, user), CHAT_MAX_INPUT_TOKENS)
        enforce_input_budget(system, user)
        with self.assertRaises(ExecutionLimitExceeded):
            enforce_input_budget("s" * 4000, "u" * 20_004)

    def test_tts_does_not_consume_text_operation_budget(self):
        begin_execution_budget(REAL_TIME_TRANSLATION)
        record_output_tokens(500, "TEXT_TO_SPEECH")
        self.assertEqual(output_limit(1200), 500)

    def test_multi_call_input_budget_uses_actual_provider_usage(self):
        begin_execution_budget(AI_CHAT_QUERY)
        self.assertEqual(current_execution_budget().remaining_input_tokens, 6000)
        enforce_input_budget("", "a" * 16000)  # estimated 4000
        record_input_tokens(4000, "TEXT_CHAT")
        self.assertEqual(current_execution_budget().remaining_input_tokens, 2000)
        enforce_input_budget("", "b" * 4000)  # estimated 1000
        record_input_tokens(1000, "TEXT_CHAT")
        self.assertEqual(current_execution_budget().remaining_input_tokens, 1000)
        with self.assertRaises(ExecutionLimitExceeded):
            enforce_input_budget("", "c" * 4004)  # estimated 1001
        enforce_input_budget("", "c" * 4000)  # estimated 1000

    def test_retry_does_not_consume_a_second_input_budget(self):
        begin_execution_budget(AI_CHAT_QUERY)
        # Preflight can be repeated for a retry; only a completed logical call
        # reports usage and consumes the operation budget.
        enforce_input_budget("", "a" * 16000)
        enforce_input_budget("", "a" * 16000)
        self.assertEqual(current_execution_budget().remaining_input_tokens, 6000)
        record_input_tokens(4000, "TEXT_CHAT")
        self.assertEqual(current_execution_budget().remaining_input_tokens, 2000)

    def test_itinerary_nested_in_chat_reuses_the_parent_budget(self):
        chat_budget = begin_execution_budget(AI_CHAT_QUERY)
        record_output_tokens(300, "TEXT_CHAT")
        nested_budget = begin_execution_budget(AI_TRIP_ITINERARY)
        self.assertIs(nested_budget, chat_budget)
        self.assertEqual(output_limit(1000), 500)


if __name__ == "__main__":
    unittest.main()

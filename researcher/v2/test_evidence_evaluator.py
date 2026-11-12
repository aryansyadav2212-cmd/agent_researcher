import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .evidence_evaluator import (
    EVALUATION_BATCH_SIZE,
    EvaluationValidationError,
    EvidenceEvaluator,
    LLMCallResult,
)
from . import client


def _contexts(count):
    return [{
        "document_id": "doc-a", "node_id": "node-a", "anchor_block_id": f"b{i}",
        "blocks": [{"block_id": f"b{i}", "text": f"Evidence {i}", "type": "paragraph"}],
    } for i in range(count)]


def _result_for(evidence):
    return json.dumps({
        "relevance": [{
            "document_id": item["document_id"], "node_id": item["node_id"],
            "block_id": item["block_id"], "classification": "supporting", "reason": "Useful.",
        } for item in evidence],
        "coverage": [],
        "sufficiency": {"status": "insufficient", "reason": "Limited evidence."},
    })


class EvidenceEvaluatorStabilityTests(unittest.TestCase):
    def test_plain_string_llm_call_remains_supported(self):
        evidence = [{"document_id": "doc-a", "node_id": "node-a", "block_id": "b0"}]
        result = EvidenceEvaluator(
            "question", _contexts(1), lambda _prompt: _result_for(evidence)
        ).evaluate()
        self.assertEqual(result["relevance"][0]["block_id"], "b0")

    def test_missing_classification_is_retried_then_fails_closed(self):
        calls = []
        def llm(prompt):
            calls.append(prompt)
            return _result_for([])
        evaluator = EvidenceEvaluator("question", _contexts(1), llm)
        with self.assertRaisesRegex(EvaluationValidationError, "did not classify every") as caught:
            evaluator.evaluate()
        self.assertEqual(len(calls), 2)
        attempts = caught.exception.diagnostics["attempts"]
        self.assertEqual([item["actual_relevance_item_count"] for item in attempts], [0, 0])
        self.assertEqual(attempts[0]["missing_evidence_ids"], [["doc-a", "node-a", "b0"]])
        self.assertEqual(attempts[0]["llm_response"]["total_tokens"], None)

    def test_generic_response_metadata_is_captured_without_openai_types(self):
        response = LLMCallResult(
            content=_result_for([]), finish_reason="length",
            prompt_tokens=120, completion_tokens=40, total_tokens=160,
        )
        with self.assertRaises(EvaluationValidationError) as caught:
            EvidenceEvaluator("question", _contexts(1), lambda _prompt: response).evaluate()
        self.assertEqual(
            caught.exception.diagnostics["attempts"][0]["llm_response"],
            {"finish_reason": "length", "prompt_tokens": 120,
             "completion_tokens": 40, "total_tokens": 160},
        )

    def test_unexpected_and_duplicate_ids_are_reported(self):
        evidence = _contexts(1)
        unexpected = json.loads(_result_for([{
            "document_id": "doc-a", "node_id": "node-a", "block_id": "other",
            "classification": "supporting", "reason": "Useful.",
        }]))
        with self.assertRaises(EvaluationValidationError) as caught:
            EvidenceEvaluator("question", evidence, lambda _prompt: json.dumps(unexpected)).evaluate()
        attempt = caught.exception.diagnostics["attempts"][0]
        self.assertEqual(attempt["unexpected_evidence_ids"], [["doc-a", "node-a", "other"]])
        self.assertEqual(attempt["missing_evidence_ids"], [["doc-a", "node-a", "b0"]])

        duplicate = json.loads(_result_for([{
            "document_id": "doc-a", "node_id": "node-a", "block_id": "b0",
            "classification": "supporting", "reason": "Useful.",
        }]))
        duplicate["relevance"].append(dict(duplicate["relevance"][0]))
        with self.assertRaises(EvaluationValidationError) as caught:
            EvidenceEvaluator("question", evidence, lambda _prompt: json.dumps(duplicate)).evaluate()
        attempt = caught.exception.diagnostics["attempts"][0]
        self.assertEqual(attempt["duplicate_evidence_ids"], [["doc-a", "node-a", "b0"]])
        self.assertEqual(attempt["actual_relevance_item_count"], 2)

    def test_sixteen_items_can_still_have_duplicate_and_missing_ids(self):
        contexts = _contexts(16)
        response = json.loads(_result_for([
            {"document_id": "doc-a", "node_id": "node-a", "block_id": f"b{i}",
             "classification": "supporting", "reason": "Useful."}
            for i in range(15)
        ] + [{
            "document_id": "doc-a", "node_id": "node-a", "block_id": "b0",
            "classification": "supporting", "reason": "Useful again.",
        }]))
        with self.assertRaises(EvaluationValidationError) as caught:
            EvidenceEvaluator("question", contexts, lambda _prompt: json.dumps(response)).evaluate()
        first = caught.exception.diagnostics["attempts"][0]
        self.assertEqual(first["expected_classification_count"], 16)
        self.assertEqual(first["actual_relevance_item_count"], 16)
        self.assertEqual(first["unique_classification_count"], 15)
        self.assertEqual(first["missing_evidence_ids"], [["doc-a", "node-a", "b15"]])
        self.assertEqual(first["duplicate_evidence_ids"], [["doc-a", "node-a", "b0"]])

    def test_both_failed_attempts_keep_separate_metadata_and_diagnostics(self):
        responses = [
            LLMCallResult(_result_for([]), "length", 100, 20, 120),
            LLMCallResult(_result_for([{
                "document_id": "doc-a", "node_id": "node-a", "block_id": "unknown",
                "classification": "supporting", "reason": "Useful.",
            }]), "stop", 110, 25, 135),
        ]
        with self.assertRaises(EvaluationValidationError) as caught:
            EvidenceEvaluator("question", _contexts(1), lambda _prompt: responses.pop(0)).evaluate()
        attempts = caught.exception.diagnostics["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual([item["llm_response"]["total_tokens"] for item in attempts], [120, 135])
        self.assertEqual(attempts[0]["missing_evidence_ids"], [["doc-a", "node-a", "b0"]])
        self.assertEqual(attempts[1]["unexpected_evidence_ids"], [["doc-a", "node-a", "unknown"]])
        self.assertIn("LLM did not classify every evidence block", attempts[0]["validation_error"])

    def test_unknown_id_and_duplicate_classification_are_rejected(self):
        for mutation, message in (("unknown", "unknown evidence ID"), ("duplicate", "Duplicate relevance")):
            with self.subTest(mutation=mutation):
                def llm(_prompt):
                    response = json.loads(_result_for([{
                        "document_id": "doc-a", "node_id": "node-a", "block_id": "b0",
                        "classification": "supporting", "reason": "Useful.",
                    }]))
                    if mutation == "unknown":
                        response["relevance"][0]["block_id"] = "missing"
                    else:
                        response["relevance"].append(dict(response["relevance"][0]))
                    return json.dumps(response)
                with self.assertRaisesRegex(ValueError, message):
                    EvidenceEvaluator("question", _contexts(1), llm).evaluate()

    def test_large_evidence_is_split_and_all_ids_are_preserved(self):
        contexts = _contexts(EVALUATION_BATCH_SIZE + 3)
        calls = []
        def llm(prompt):
            calls.append(prompt)
            start = prompt.index("EVIDENCE BLOCKS:\n") + len("EVIDENCE BLOCKS:\n")
            end = prompt.index("\n\nPerform THREE", start)
            return _result_for(json.loads(prompt[start:end]))
        result = EvidenceEvaluator("question", contexts, llm).evaluate()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result["relevance"]), len(contexts))

    def test_provider_timeout_and_error_are_not_swallowed(self):
        for failure in (TimeoutError("provider timeout"), RuntimeError("provider error")):
            with self.subTest(failure=type(failure).__name__):
                class FakeOpenAI:
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs
                        self.chat = self
                        self.completions = self
                    def create(self, **_kwargs):
                        raise failure
                with patch("openai.OpenAI", FakeOpenAI):
                    with self.assertRaises(client.EvaluationProviderError):
                        client._openai_evaluator_call("safe prompt")

    def test_openai_adapter_returns_only_generic_content_and_usage_metadata(self):
        captured = {}
        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client_options"] = kwargs
                self.chat = self
                self.completions = self
            def create(self, **kwargs):
                captured["request"] = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"relevance": []}'),
                    )],
                    usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8, total_tokens=29),
                )
        with patch("openai.OpenAI", FakeOpenAI):
            result = client._openai_evaluator_call("opaque prompt")
        self.assertEqual(result, LLMCallResult(
            content='{"relevance": []}', finish_reason="stop",
            prompt_tokens=21, completion_tokens=8, total_tokens=29,
        ))
        self.assertEqual(captured["client_options"], {"timeout": 45.0, "max_retries": 0})
        self.assertEqual(captured["request"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()

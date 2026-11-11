
import json

from evidence_evaluator import EvidenceEvaluator


# --------------------------------------------------
# 1. Mock LLM response
# --------------------------------------------------

MOCK_RESPONSE = {
    "relevance": [
        {
            "node_id": "node_1",
            "block_id": "blk_1",
            "classification": "relevant",
            "reason": "Directly explains the alternative."
        },
        {
            "node_id": "node_1",
            "block_id": "blk_2",
            "classification": "supporting",
            "reason": "Provides useful background."
        },
        {
            "node_id": "node_2",
            "block_id": "blk_3",
            "classification": "irrelevant",
            "reason": "Unrelated information."
        }
    ],
    "coverage": [
        {
            "requirement": "Identify an alternative to RAG",
            "status": "supported",
            "evidence": [
                {"node_id": "node_1", "block_id": "blk_1"}
            ]
        },
        {
            "requirement": "Explain its advantages over RAG",
            "status": "partial",
            "evidence": [
                {"node_id": "node_1", "block_id": "blk_2"}
            ]
        }
    ],
    "sufficiency": {
        "status": "insufficient",
        "reason": "The comparison is incomplete."
    }
}


# --------------------------------------------------
# 2. Mock LLM callable
# --------------------------------------------------

class MockLLM:

    def __init__(self, response=None):
        self.response = response or MOCK_RESPONSE
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1

        print("\nMock LLM received prompt.")
        print(f"Prompt length: {len(prompt)} characters")

        return json.dumps(self.response)


# --------------------------------------------------
# 3. Sample expanded contexts
# --------------------------------------------------

expanded_contexts = [
    {
        "node_id": "node_1",
        "anchor_block_id": "blk_1",
        "blocks": [
            {
                "block_id": "blk_1",
                "type": "paragraph",
                "text": "PageIndex introduces an alternative to conventional RAG.",
                "start_line": 10,
                "end_line": 12,
                "is_anchor": True
            },
            {
                "block_id": "blk_2",
                "type": "paragraph",
                "text": "Traditional RAG commonly uses vector similarity retrieval.",
                "start_line": 13,
                "end_line": 15,
                "is_anchor": False
            }
        ]
    },
    {
        "node_id": "node_1",
        "anchor_block_id": "blk_2",
        "blocks": [
            {
                "block_id": "blk_1",
                "type": "paragraph",
                "text": "PageIndex introduces an alternative to conventional RAG.",
                "start_line": 10,
                "end_line": 12,
                "is_anchor": False
            },
            {
                "block_id": "blk_2",
                "type": "paragraph",
                "text": "Traditional RAG commonly uses vector similarity retrieval.",
                "start_line": 13,
                "end_line": 15,
                "is_anchor": True
            }
        ]
    },
    {
        "node_id": "node_2",
        "anchor_block_id": "blk_3",
        "blocks": [
            {
                "block_id": "blk_3",
                "type": "paragraph",
                "text": "This paragraph discusses an unrelated benchmark.",
                "start_line": 30,
                "end_line": 32,
                "is_anchor": True
            }
        ]
    }
]


# --------------------------------------------------
# 4. Main tests
# --------------------------------------------------

def test_prepare_evidence():

    evaluator = EvidenceEvaluator(
        query="What is an alternative to RAG?",
        expanded_contexts=expanded_contexts,
        llm_call=MockLLM()
    )

    result = evaluator.prepare_evidence()

    assert result["total_unique_blocks"] == 3

    assert len(result["evidence"]) == 3

    # blk_1 appeared in two contexts.
    blk_1 = next(
        item for item in result["evidence"]
        if item["block_id"] == "blk_1"
    )

    assert len(blk_1["retrieved_from"]) == 2
    assert blk_1["is_anchor"] is True

    print("PASS: prepare_evidence")


def test_evaluation_methods():

    mock_llm = MockLLM()

    evaluator = EvidenceEvaluator(
        query="What is an alternative to RAG?",
        expanded_contexts=expanded_contexts,
        llm_call=mock_llm
    )

    relevance = evaluator.evaluate_relevance()

    assert len(relevance) == 3

    classifications = {
        item["block_id"]: item["classification"]
        for item in relevance
    }

    assert classifications["blk_1"] == "relevant"
    assert classifications["blk_2"] == "supporting"
    assert classifications["blk_3"] == "irrelevant"

    print("PASS: evaluate_relevance")

    coverage = evaluator.evaluate_coverage()

    assert len(coverage) == 2
    assert coverage[0]["status"] == "supported"
    assert coverage[1]["status"] == "partial"

    print("PASS: evaluate_coverage")

    sufficiency = evaluator.evaluate_sufficiency()

    assert sufficiency["status"] == "insufficient"

    print("PASS: evaluate_sufficiency")

    # All three methods should share one LLM evaluation.
    assert mock_llm.call_count == 1

    print("PASS: LLM called only once")


def test_invalid_evidence_id():

    invalid_response = json.loads(json.dumps(MOCK_RESPONSE))

    invalid_response["relevance"][0]["block_id"] = "unknown_block"

    evaluator = EvidenceEvaluator(
        query="Test query",
        expanded_contexts=expanded_contexts,
        llm_call=MockLLM(invalid_response)
    )

    try:
        evaluator.evaluate()
        assert False, "Expected ValueError was not raised."

    except ValueError:
        print("PASS: Unknown evidence ID rejected")


def test_missing_classification():

    invalid_response = json.loads(json.dumps(MOCK_RESPONSE))

    invalid_response["relevance"].pop()

    evaluator = EvidenceEvaluator(
        query="Test query",
        expanded_contexts=expanded_contexts,
        llm_call=MockLLM(invalid_response)
    )

    try:
        evaluator.evaluate()
        assert False, "Expected ValueError was not raised."

    except ValueError:
        print("PASS: Missing classification rejected")


def test_invalid_classification():

    invalid_response = json.loads(json.dumps(MOCK_RESPONSE))

    invalid_response["relevance"][0]["classification"] = "maybe"

    evaluator = EvidenceEvaluator(
        query="Test query",
        expanded_contexts=expanded_contexts,
        llm_call=MockLLM(invalid_response)
    )

    try:
        evaluator.evaluate()
        assert False, "Expected ValueError was not raised."

    except ValueError:
        print("PASS: Invalid classification rejected")


def test_invalid_sufficiency():

    invalid_response = json.loads(json.dumps(MOCK_RESPONSE))

    invalid_response["sufficiency"]["status"] = "maybe"

    evaluator = EvidenceEvaluator(
        query="Test query",
        expanded_contexts=expanded_contexts,
        llm_call=MockLLM(invalid_response)
    )

    try:
        evaluator.evaluate()
        assert False, "Expected ValueError was not raised."

    except ValueError:
        print("PASS: Invalid sufficiency rejected")


def test_empty_evidence():

    mock_llm = MockLLM()

    evaluator = EvidenceEvaluator(
        query="Test query",
        expanded_contexts=[],
        llm_call=mock_llm
    )

    result = evaluator.evaluate()

    assert result["relevance"] == []
    assert result["coverage"] == []
    assert result["sufficiency"]["status"] == "insufficient"

    assert mock_llm.call_count == 0

    print("PASS: Empty evidence handled without LLM call")


# --------------------------------------------------
# 5. Run everything
# --------------------------------------------------

if __name__ == "__main__":

    test_prepare_evidence()
    test_evaluation_methods()
    test_invalid_evidence_id()
    test_missing_classification()
    test_invalid_classification()
    test_invalid_sufficiency()
    test_empty_evidence()

    print("\nAll tests passed!")

import json
import time
from dataclasses import dataclass
from typing import Any


EVALUATION_BATCH_SIZE = 20
EVALUATION_RETRIES = 1
EVALUATION_TIME_BUDGET_SECONDS = 120.0


@dataclass(frozen=True)
class LLMCallResult:
    """Provider-neutral model response content and safe response metadata."""

    content: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class EvaluationValidationError(ValueError):
    """Validation failure with JSON-safe provenance diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics

class EvidenceEvaluator:
    def __init__(
        self,
        query: str,
        expanded_contexts: list[dict],
        llm_call
    ):
        self.query = query
        self.expanded_contexts = expanded_contexts
        self.llm_call = llm_call

        self._prepared_evidence = None
        self._evaluation = None

    def prepare_evidence(self):
        unique_evidence = {}

        for context_index, context in enumerate(self.expanded_contexts):

            node_id = context.get("node_id")
            document_id = context.get("document_id") or "__legacy__"
            anchor_block_id = context.get("anchor_block_id")

            if not node_id:
                raise ValueError(
                    f"Context at index {context_index} is missing node_id."
                )

            blocks = context.get("blocks")

            if not isinstance(blocks, list):
                raise ValueError(
                    f"Context at index {context_index} must contain a blocks list."
                )

            for block in blocks:

                block_id = block.get("block_id")

                if not block_id:
                    raise ValueError(
                        f"Context at index {context_index} "
                        "contains a block without block_id."
                    )

                key = (document_id, node_id, block_id)

                if key not in unique_evidence:
                    unique_evidence[key] = {
                        "document_id": document_id,
                        "node_id": node_id,
                        "block_id": block_id,
                        "type": block.get("type"),
                        "text": block.get("text"),
                        "start_line": block.get("start_line"),
                        "end_line": block.get("end_line"),
                        "is_anchor": bool(block.get("is_anchor", False)),
                        "retrieved_from": []
                    }

                evidence = unique_evidence[key]

                source = {
                    "context_index": context_index,
                    "anchor_block_id": anchor_block_id
                }

                if source not in evidence["retrieved_from"]:
                    evidence["retrieved_from"].append(source)

                if block.get("is_anchor", False):
                    evidence["is_anchor"] = True

        self._prepared_evidence = {
            "query": self.query,
            "total_unique_blocks": len(unique_evidence),
            "evidence": list(unique_evidence.values())
        }

        return self._prepared_evidence

    def _run_evaluation(self):
        """
        Run all three evaluations in one LLM call.
        Cache the result to avoid repeated API calls.
        """

        if self._evaluation is not None:
            return self._evaluation

        prepared = self._prepared_evidence or self.prepare_evidence()

        evidence = prepared["evidence"]

        if not evidence:
            self._evaluation = {
                "relevance": [],
                "coverage": [],
                "sufficiency": {
                    "status": "insufficient",
                    "reason": "No evidence blocks were retrieved."
                }
            }
            return self._evaluation

        all_relevance = []
        all_coverage = []
        sufficiency_results = []
        started = time.monotonic()
        for offset in range(0, len(evidence), EVALUATION_BATCH_SIZE):
            if time.monotonic() - started >= EVALUATION_TIME_BUDGET_SECONDS:
                raise TimeoutError("Evidence evaluation exceeded its 120-second budget.")
            batch = evidence[offset:offset + EVALUATION_BATCH_SIZE]
            result = self._evaluate_batch(batch)
            all_relevance.extend(result["relevance"])
            all_coverage.extend(result["coverage"])
            sufficiency_results.append(result["sufficiency"])

        statuses = [item["status"] for item in sufficiency_results]
        self._evaluation = {
            "relevance": all_relevance,
            "coverage": all_coverage,
            "sufficiency": {
                "status": "sufficient" if all(s == "sufficient" for s in statuses) else "insufficient",
                "reason": "All evidence batches were sufficient." if all(s == "sufficient" for s in statuses)
                else "At least one evidence batch was insufficient.",
            },
        }
        return self._evaluation

    def _evaluate_batch(self, evidence):
        prompt = f"""
You are an evidence evaluator for a research paper retrieval system.

Your job is to evaluate retrieved evidence against a user's question.

USER QUESTION:
{self.query}

EVIDENCE BLOCKS:
{json.dumps(evidence, ensure_ascii=False)}

Perform THREE evaluations.

1. BLOCK-LEVEL RELEVANCE

Classify every evidence block as exactly one of:

- relevant: directly answers or provides important information for the question.
- supporting: provides useful background or indirect support.
- irrelevant: does not meaningfully contribute to answering the question.

Give a short reason for each classification.

Every evidence block must receive exactly one classification.
For each classification, include the evidence block's document_id, node_id,
and block_id exactly as supplied.

2. QUERY COVERAGE

Identify the major information requirements needed to answer the question.

For each requirement, assign exactly one status:

- supported: evidence adequately answers this requirement.
- partial: evidence provides some information but is incomplete.
- missing: retrieved evidence does not answer this requirement.

Include the document_id, node_id, and block_id of each evidence block
supporting a requirement.

Do not invent evidence IDs.

3. OVERALL SUFFICIENCY

Determine whether the entire evidence collection is sufficient to answer
the user's question accurately.

Use exactly one status:

- sufficient
- insufficient

Consider missing and partially supported requirements.

Do not assume that high relevance means sufficient evidence.

IMPORTANT RULES:

- Evaluate only the supplied evidence.
- Do not use external knowledge.
- Do not invent facts or evidence.
- Do not treat a document's navigation summary as original source evidence.
- If the evidence cannot establish a comparison, mark that requirement partial
  or missing.
- Keep reasons concise.

Return ONLY valid JSON using this exact structure:

{{
    "relevance": [
        {{
            "document_id": "string",
            "node_id": "string",
            "block_id": "string",
            "classification": "relevant",
            "reason": "string"
        }}
    ],
    "coverage": [
        {{
            "requirement": "string",
            "status": "supported",
            "evidence": [
                {{
                    "document_id": "string",
                    "node_id": "string",
                    "block_id": "string"
                }}
            ]
        }}
    ],
    "sufficiency": {{
        "status": "sufficient",
        "reason": "string"
    }}
}}
"""

        last_error = None
        failed_attempts = []
        for attempt in range(EVALUATION_RETRIES + 1):
            response_metadata = {
                "finish_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            retry_note = ""
            if attempt and last_error:
                retry_note = ("\nYour previous response failed validation: "
                              f"{last_error}. Return exactly one classification for every supplied ID, "
                              "with no other IDs and no duplicates.")
            try:
                response = self.llm_call(prompt + retry_note)
                if isinstance(response, LLMCallResult):
                    content = response.content
                    response_metadata = response.metadata()
                elif isinstance(response, str):
                    content = response
                    response_metadata = {
                        "finish_reason": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    }
                else:
                    raise TypeError("LLM call must return a string or LLMCallResult.")

                try:
                    result = json.loads(content)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise EvaluationValidationError(
                        "LLM returned invalid JSON for evidence evaluation.",
                        self._empty_validation_diagnostics(evidence),
                    ) from exc

                self._validate_evaluation(result, evidence)
                return result
            except (ValueError, TypeError) as exc:
                last_error = str(exc)
                validation_diagnostics = getattr(
                    exc, "diagnostics", self._empty_validation_diagnostics(evidence)
                )
                failed_attempts.append({
                    "attempt": attempt + 1,
                    **validation_diagnostics,
                    "llm_response": response_metadata,
                    "validation_error": last_error,
                })
                if attempt == EVALUATION_RETRIES:
                    raise EvaluationValidationError(
                        f"Evidence evaluation failed after {attempt + 1} attempt(s): {last_error}",
                        {"attempts": failed_attempts},
                    ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _empty_validation_diagnostics(evidence):
        return {
            "expected_classification_count": len(evidence),
            "actual_relevance_item_count": None,
            "unique_classification_count": 0,
            "missing_evidence_ids": [
                [item["document_id"], item["node_id"], item["block_id"]]
                for item in evidence
            ],
            "unexpected_evidence_ids": [],
            "duplicate_evidence_ids": [],
        }

    def _validate_evaluation(self, result, evidence):
        """
        Validate LLM output against actual retrieved evidence.
        """

        valid_ids = {
            (item["document_id"], item["node_id"], item["block_id"])
            for item in evidence
        }

        diagnostics = self._empty_validation_diagnostics(evidence)

        def invalid(message):
            raise EvaluationValidationError(
                message,
                {**diagnostics, "validation_error": message},
            )

        if not isinstance(result, dict):
            invalid("Evaluation must be a JSON object.")

        # Validate relevance.
        relevance = result.get("relevance")

        if not isinstance(relevance, list):
            invalid("Invalid relevance evaluation.")

        counts = {}
        malformed_relevance = None

        for index, item in enumerate(relevance):
            if not isinstance(item, dict):
                malformed_relevance = f"Relevance item at index {index} must be a dictionary."
                continue

            for field in ("document_id", "node_id", "block_id"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    malformed_relevance = (
                        f"Relevance item at index {index} must contain a "
                        f"non-empty string {field}."
                    )
                    break
            else:
                key = (
                    item.get("document_id"),
                    item.get("node_id"),
                    item.get("block_id"),
                )
                counts[key] = counts.get(key, 0) + 1

        observed_ids = set(counts)
        missing_ids = valid_ids - observed_ids
        unexpected_ids = observed_ids - valid_ids
        duplicate_ids = {key for key, count in counts.items() if count > 1}
        diagnostics.update({
            "actual_relevance_item_count": len(relevance),
            "unique_classification_count": len(observed_ids),
            "missing_evidence_ids": [list(key) for key in sorted(missing_ids)],
            "unexpected_evidence_ids": [list(key) for key in sorted(unexpected_ids)],
            "duplicate_evidence_ids": [list(key) for key in sorted(duplicate_ids)],
        })

        if malformed_relevance:
            invalid(malformed_relevance)
        if unexpected_ids:
            invalid(f"LLM returned unknown evidence ID: {sorted(unexpected_ids)[0]}")
        if duplicate_ids:
            invalid(f"Duplicate relevance evaluation: {sorted(duplicate_ids)[0]}")
        if missing_ids:
            invalid("LLM did not classify every evidence block.")
        for index, item in enumerate(relevance):
            if item.get("classification") not in {
                "relevant",
                "supporting",
                "irrelevant"
            }:
                invalid("Invalid relevance classification.")

        # Validate coverage.
        coverage = result.get("coverage")

        if not isinstance(coverage, list):
            invalid("Invalid coverage evaluation.")

        for requirement_index, requirement in enumerate(coverage):
            if not isinstance(requirement, dict):
                invalid(
                    f"Coverage item at index {requirement_index} must be a dictionary."
                )

            if requirement.get("status") not in {
                "supported",
                "partial",
                "missing"
            }:
                invalid(
                    f"Invalid coverage status at index {requirement_index}."
                )

            coverage_evidence = requirement.get("evidence")
            if not isinstance(coverage_evidence, list):
                invalid(
                    f"Coverage evidence at index {requirement_index} must be a list."
                )

            for evidence_index, item in enumerate(coverage_evidence):
                if not isinstance(item, dict):
                    invalid(
                        f"Coverage evidence item at requirement index "
                        f"{requirement_index}, evidence index {evidence_index} "
                        "must be a dictionary."
                    )

                for field in ("document_id", "node_id", "block_id"):
                    value = item.get(field)
                    if not isinstance(value, str) or not value.strip():
                        invalid(
                            f"Coverage evidence item at requirement index "
                            f"{requirement_index}, evidence index {evidence_index} "
                            f"must contain a non-empty string {field}."
                        )

                key = (
                    item.get("document_id"),
                    item.get("node_id"),
                    item.get("block_id"),
                )

                if key not in valid_ids:
                    invalid(
                        f"Coverage references unknown evidence: {key}"
                    )

        # Validate sufficiency.
        sufficiency = result.get("sufficiency")

        if not isinstance(sufficiency, dict):
            invalid("Invalid sufficiency evaluation.")

        if sufficiency.get("status") not in {
            "sufficient",
            "insufficient"
        }:
            invalid("Invalid sufficiency status.")

    def evaluate_relevance(self):
        return self._run_evaluation()["relevance"]

    def evaluate_coverage(self):
        return self._run_evaluation()["coverage"]

    def evaluate_sufficiency(self):
        return self._run_evaluation()["sufficiency"]

    def evaluate(self):
        """
        Return the complete evaluation.
        """
        return self._run_evaluation()

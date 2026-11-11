import json

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

                key = (node_id, block_id)

                if key not in unique_evidence:
                    unique_evidence[key] = {
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

2. QUERY COVERAGE

Identify the major information requirements needed to answer the question.

For each requirement, assign exactly one status:

- supported: evidence adequately answers this requirement.
- partial: evidence provides some information but is incomplete.
- missing: retrieved evidence does not answer this requirement.

Include the node_id and block_id of the evidence supporting each requirement.

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

        raw_response = self.llm_call(prompt)

        try:
            result = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "LLM returned invalid JSON for evidence evaluation."
            ) from exc

        self._validate_evaluation(result, evidence)

        self._evaluation = result

        return result

    def _validate_evaluation(self, result, evidence):
        """
        Validate LLM output against actual retrieved evidence.
        """

        if not isinstance(result, dict):
            raise ValueError("Evaluation must be a JSON object.")

        valid_ids = {
            (item["node_id"], item["block_id"])
            for item in evidence
        }

        # Validate relevance.
        relevance = result.get("relevance")

        if not isinstance(relevance, list):
            raise ValueError("Invalid relevance evaluation.")

        seen_ids = set()

        for item in relevance:

            key = (
                item.get("node_id"),
                item.get("block_id")
            )

            if key not in valid_ids:
                raise ValueError(
                    f"LLM returned unknown evidence ID: {key}"
                )

            if key in seen_ids:
                raise ValueError(
                    f"Duplicate relevance evaluation: {key}"
                )

            if item.get("classification") not in {
                "relevant",
                "supporting",
                "irrelevant"
            }:
                raise ValueError("Invalid relevance classification.")

            seen_ids.add(key)

        if seen_ids != valid_ids:
            raise ValueError(
                "LLM did not classify every evidence block."
            )

        # Validate coverage.
        coverage = result.get("coverage")

        if not isinstance(coverage, list):
            raise ValueError("Invalid coverage evaluation.")

        for requirement in coverage:

            if requirement.get("status") not in {
                "supported",
                "partial",
                "missing"
            }:
                raise ValueError("Invalid coverage status.")

            for item in requirement.get("evidence", []):

                key = (
                    item.get("node_id"),
                    item.get("block_id")
                )

                if key not in valid_ids:
                    raise ValueError(
                        f"Coverage references unknown evidence: {key}"
                    )

        # Validate sufficiency.
        sufficiency = result.get("sufficiency")

        if not isinstance(sufficiency, dict):
            raise ValueError("Invalid sufficiency evaluation.")

        if sufficiency.get("status") not in {
            "sufficient",
            "insufficient"
        }:
            raise ValueError("Invalid sufficiency status.")

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

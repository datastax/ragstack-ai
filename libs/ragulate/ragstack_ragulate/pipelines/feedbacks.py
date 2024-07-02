from typing import Any, Dict, List

import numpy as np
from trulens_eval import Feedback
from trulens_eval.app import App
from trulens_eval.feedback import GroundTruthAgreement
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.utils.serial import Lens


class Feedbacks:
    _context: Lens
    _llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider, pipeline: Any) -> None:
        self._context = App.select_context(pipeline)
        self._llm_provider = llm_provider

    def groundedness(self) -> Feedback:
        return (
            Feedback(
                self._llm_provider.groundedness_measure_with_cot_reasons,
                name="groundedness",
            )
            .on(self._context.collect())  # collect context chunks into a list
            .on_output()
        )

    def answer_relevance(self) -> Feedback:
        return Feedback(
            self._llm_provider.relevance_with_cot_reasons, name="answer_relevance"
        ).on_input_output()

    def context_relevance(self) -> Feedback:
        return (
            Feedback(
                self._llm_provider.qs_relevance_with_cot_reasons,
                name="context_relevance",
            )
            .on_input()
            .on(self._context)
            .aggregate(np.mean)
        )

    def answer_correctness(self, golden_set: List[Dict[str, str]]) -> Feedback:
        # GroundTruth for comparing the Answer to the Ground-Truth Answer
        ground_truth_collection = GroundTruthAgreement(
            ground_truth=golden_set, provider=self._llm_provider
        )
        return Feedback(
            ground_truth_collection.agreement_measure, name="answer_correctness"
        ).on_input_output()

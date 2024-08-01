# ruff: noqa: T201
from __future__ import annotations

import random
import signal
import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm
from trulens_eval import Tru, TruChain
from trulens_eval.feedback.provider import AzureOpenAI, Huggingface, LLMProvider, OpenAI
from trulens_eval.schema.feedback import FeedbackMode, FeedbackResultStatus
from typing_extensions import Never, override

from ragstack_ragulate.logging_config import logger
from ragstack_ragulate.utils import get_tru

from .base_pipeline import BasePipeline
from .feedbacks import Feedbacks

if TYPE_CHECKING:
    from ragstack_ragulate.datasets import BaseDataset, QueryItem


class QueryPipeline(BasePipeline):
    """Query pipeline."""

    _sigint_received = False

    _tru: Tru
    _name: str
    _progress: tqdm[Never]
    _query_items: dict[str, list[QueryItem]]
    _golden_sets: dict[str, list[dict[str, str]]]
    _total_queries: int = 0
    _total_feedbacks: int = 0
    _finished_feedbacks: int = 0
    _finished_queries: int = 0
    _evaluation_running = False

    @property
    @override
    def pipeline_type(self) -> str:
        return "query"

    @property
    @override
    def get_reserved_params(self) -> list[str]:
        return []

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        ingredients: dict[str, Any],
        datasets: list[BaseDataset],
        sample_percent: float = 1.0,
        random_seed: int | None = None,
        restart_pipeline: bool = False,
        llm_provider: str = "OpenAI",
        model_name: str | None = None,
    ):
        self._query_items = {}
        self._golden_sets = {}
        super().__init__(
            recipe_name=recipe_name,
            script_path=script_path,
            method_name=method_name,
            ingredients=ingredients,
            datasets=datasets,
        )

        self.sample_percent = sample_percent
        self.random_seed = random_seed
        self.restart_pipeline = restart_pipeline
        self.llm_provider = llm_provider
        self.model_name = model_name

        # Set up the signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.signal_handler)

        self._tru = get_tru(recipe_name=self.recipe_name)
        if self.restart_pipeline:
            # TODO: Work with TruLens to get a new method added
            # so we can just delete a single "app" instead of the whole
            # database.
            self._tru.reset_database()

        total_existing_queries = 0
        for dataset in datasets:
            query_items = dataset.get_query_items()
            if self.sample_percent < 1.0:
                if self.random_seed is not None:
                    random.seed(self.random_seed)
                sampled_indices = random.sample(
                    range(len(query_items)), int(self.sample_percent * len(query_items))
                )
                query_items = [query_items[i] for i in sampled_indices]

            # Check for existing records and filter queries
            existing_records, _feedbacks = self._tru.get_records_and_feedback(
                app_ids=[dataset.name]
            )
            existing_queries = existing_records["input"].dropna().tolist()
            total_existing_queries += len(existing_queries)

            query_items = [
                query_item
                for query_item in query_items
                if query_item.query not in existing_queries
            ]

            self._query_items[dataset.name] = query_items
            self._golden_sets[dataset.name] = dataset.get_golden_set()
            self._total_queries += len(self._query_items[dataset.name])

        metric_count = 4
        self._total_feedbacks = self._total_queries * metric_count

        # Set finished queries count to total existing queries
        self._finished_queries = total_existing_queries

    def signal_handler(self, _: Any, __: Any) -> None:
        """Handle SIGINT signal."""
        self._sigint_received = True
        self.stop_evaluation("sigint")

    def start_evaluation(self) -> None:
        """Start evaluation."""
        self._tru.start_evaluator(disable_tqdm=True)
        self._evaluation_running = True

    def export_results(self) -> None:
        """Export results."""
        for dataset_name in self._query_items:
            records, _feedback_names = self._tru.get_records_and_feedback(
                app_ids=[dataset_name]
            )

            # Export to JSON
            records.to_json(f"{self._name}_{dataset_name}_results.json")

    def stop_evaluation(self, loc: str) -> None:
        """Stop evaluation."""
        if self._evaluation_running:
            try:
                logger.debug(f"Stopping evaluation from: {loc}")
                self._tru.stop_evaluator()
                self._evaluation_running = False
                self._tru.delete_singleton()
            except Exception:  # noqa: BLE001
                logger.exception("issue stopping evaluator")
            finally:
                self._progress.close()
                self.export_results()

    def update_progress(self, query_change: int = 0) -> None:
        """Update progress bar."""
        self._finished_queries += query_change

        status = self._tru.db.get_feedback_count_by_status()
        done = status.get(FeedbackResultStatus.DONE, 0)

        postfix = {
            "q": self._finished_queries,
            "d": done,
            "r": status.get(FeedbackResultStatus.RUNNING, 0),
            "w": status.get(FeedbackResultStatus.NONE, 0),
            "f": status.get(FeedbackResultStatus.FAILED, 0),
            "s": status.get(FeedbackResultStatus.SKIPPED, 0),
        }
        self._progress.set_postfix(postfix)

        update = query_change + (done - self._finished_feedbacks)
        if update > 0:
            self._progress.update(update)

        self._finished_feedbacks = done

    def get_provider(self) -> LLMProvider:
        """Get the LLM provider."""
        llm_provider = self.llm_provider.lower()
        model_name = self.model_name

        if llm_provider == "openai":
            return OpenAI(model_engine=model_name)
        if llm_provider == "azureopenai":
            return AzureOpenAI(deployment_name=model_name)
        if llm_provider == "huggingface":
            return Huggingface(name=model_name)
        raise ValueError(f"Unsupported provider: {llm_provider}")

    def query(self) -> None:
        """Run the query pipeline."""
        query_method = self.get_method()

        pipeline = query_method(**self.ingredients)
        llm_provider = self.get_provider()

        print(f"Using provider: {llm_provider} for evaluation.")

        feedbacks = Feedbacks(llm_provider=llm_provider, pipeline=pipeline)

        self.start_evaluation()

        time.sleep(0.1)
        logger.info(
            f"Starting query {self.recipe_name} "
            f"on {self.script_path}/{self.method_name} "
            f"with ingredients: {self.ingredients} "
            f"on datasets: {self.dataset_names()}"
        )
        logger.info(
            "Progress postfix legend: (q)ueries completed; Evaluations (d)one, "
            "(r)unning, (w)aiting, (f)ailed, (s)kipped"
        )

        self._progress = tqdm(
            total=(self._total_queries + self._total_feedbacks),
            initial=self._finished_queries,
        )

        for dataset_name in self._query_items:
            feedback_functions = [
                feedbacks.answer_correctness(
                    golden_set=self._golden_sets[dataset_name]
                ),
                feedbacks.answer_relevance(),
                feedbacks.context_relevance(),
                feedbacks.groundedness(),
            ]

            recorder = TruChain(
                pipeline,
                app_id=dataset_name,
                feedbacks=feedback_functions,
                feedback_mode=FeedbackMode.DEFERRED,
            )

            for query_item in self._query_items[dataset_name]:
                if self._sigint_received:
                    break
                try:
                    with recorder as recording:
                        recording.record_metadata = query_item.metadata
                        pipeline.invoke(query_item.query)
                except Exception as e:  # noqa: BLE001
                    err = f"Query: '{query_item.query}' caused exception, skipping."
                    logger.exception(err)
                    # TODO: figure out why the logger isn't working after tru-lens starts. For now use print().  # noqa: E501
                    print(f"{err} Exception {e}")
                finally:
                    self.update_progress(query_change=1)

        while self._finished_feedbacks < self._total_feedbacks:
            if self._sigint_received:
                break
            self.update_progress()
            time.sleep(1)

        self.stop_evaluation(loc="end")

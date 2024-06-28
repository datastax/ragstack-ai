import random
import signal
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from trulens_eval import Tru, TruChain
from trulens_eval.feedback.provider import (
    AzureOpenAI,
    Bedrock,
    Huggingface,
    Langchain,
    LiteLLM,
    OpenAI,
)
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.schema.feedback import FeedbackMode, FeedbackResultStatus

from ragstack_ragulate.datasets import BaseDataset

from ..logging_config import logger
from ..utils import get_tru
from .base_pipeline import BasePipeline
from .feedbacks import Feedbacks


class QueryPipeline(BasePipeline):
    _sigint_received = False

    _tru: Tru
    _name: str
    _progress: tqdm
    _queries: Dict[str, List[str]] = {}
    _golden_sets: Dict[str, List[Dict[str, str]]] = {}
    _total_queries: int = 0
    _total_feedbacks: int = 0
    _finished_feedbacks: int = 0
    _finished_queries: int = 0
    _evaluation_running = False

    @property
    def PIPELINE_TYPE(self):
        return "query"

    @property
    def get_reserved_params(self) -> List[str]:
        return []

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        ingredients: Dict[str, Any],
        datasets: List[BaseDataset],
        sample_percent: float = 1.0,
        random_seed: Optional[int] = None,
        restart_pipeline: Optional[bool] = False,
        llm_provider: Optional[str] = "OpenAI",
        model_name: Optional[str] = None,
        **kwargs,
    ):
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

        for dataset in datasets:
            queries, golden_set = dataset.get_queries_and_golden_set()
            if self.sample_percent < 1.0:
                if self.random_seed is not None:
                    random.seed(self.random_seed)
                sampled_indices = random.sample(
                    range(len(queries)), int(self.sample_percent * len(queries))
                )
                queries = [queries[i] for i in sampled_indices]

            # Check for existing records and filter queries
            existing_records, _feedbacks = self._tru.get_records_and_feedback(
                app_ids=[dataset.name]
            )
            existing_queries = existing_records["input"].dropna().tolist()
            queries = [query for query in queries if query not in existing_queries]

            self._queries[dataset.name] = queries
            self._golden_sets[dataset.name] = golden_set
            self._total_queries += len(self._queries[dataset.name])

        metric_count = 4
        self._total_feedbacks = self._total_queries * metric_count

    def signal_handler(self, sig, frame):
        self._sigint_received = True
        self.stop_evaluation("sigint")

    def start_evaluation(self):
        self._tru.start_evaluator(disable_tqdm=True)
        self._evaluation_running = True

    def stop_evaluation(self, loc: str):
        if self._evaluation_running:
            try:
                logger.debug(f"Stopping evaluation from: {loc}")
                self._tru.stop_evaluator()
                self._evaluation_running = False
                self._tru.delete_singleton()
            except Exception as e:
                logger.error(f"issue stopping evaluator: {e}")
            finally:
                self._progress.close()

    def update_progress(self, query_change: int = 0):
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
        llm_provider = self.llm_provider.lower()
        model_name = self.model_name

        if llm_provider == "openai":
            return OpenAI(model_engine=model_name)
        elif llm_provider == "azureopenai":
            return AzureOpenAI(deployment_name=model_name)
        elif llm_provider == "bedrock":
            return Bedrock(model_id=model_name)
        elif llm_provider == "litellm":
            return LiteLLM(model_engine=model_name)
        elif llm_provider == "Langchain":
            return Langchain(model_engine=model_name)
        elif llm_provider == "huggingface":
            return Huggingface(name=model_name)
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")

    def query(self):
        query_method = self.get_method()

        pipeline = query_method(**self.ingredients)
        llm_provider = self.get_provider()

        feedbacks = Feedbacks(llm_provider=llm_provider, pipeline=pipeline)

        self.start_evaluation()

        time.sleep(0.1)
        logger.info(
            f"Starting query {self.recipe_name} on {self.script_path}/{self.method_name} with ingredients: {self.ingredients} on datasets: {self.dataset_names()}"
        )
        logger.info(
            "Progress postfix legend: (q)ueries completed; Evaluations (d)one, (r)unning, (w)aiting, (f)ailed, (s)kipped"
        )

        self._progress = tqdm(total=(self._total_queries + self._total_feedbacks))

        for dataset_name in self._queries:
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

            for query in self._queries[dataset_name]:
                if self._sigint_received:
                    break
                try:
                    with recorder:
                        pipeline.invoke(query)
                except Exception as e:
                    # TODO: figure out why the logger isn't working after tru-lens starts. For now use print()
                    print(
                        f"ERROR: Query: '{query}' caused exception, skipping. Exception {e}"
                    )
                    logger.error(f"Query: '{query}' caused exception: {e}, skipping.")
                finally:
                    self.update_progress(query_change=1)

        while self._finished_feedbacks < self._total_feedbacks:
            if self._sigint_received:
                break
            self.update_progress()
            time.sleep(1)

        self.stop_evaluation(loc="end")

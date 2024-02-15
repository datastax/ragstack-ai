import logging
import time
from dataclasses import dataclass
from typing import Callable, Any, List

from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.schemas import Dataset

LANGSMITH_CLIENT = Client()


def _create_langsmith_dataset(name: str) -> Dataset:
    if not LANGSMITH_CLIENT.has_dataset(dataset_name=name):
        try:
            return LANGSMITH_CLIENT.create_dataset(name)
        except Exception as e:
            logging.error(f"Failed to create dataset {name}: {e}")
            if LANGSMITH_CLIENT.has_dataset(dataset_name=name):
                logging.info(f"Dataset {name} was created by another run")
                return LANGSMITH_CLIENT.read_dataset(dataset_name=name)
            raise e
    return LANGSMITH_CLIENT.read_dataset(dataset_name=name)


def ensure_langsmith_dataset(name: str, input: Any, output: Any) -> None:
    dataset = _create_langsmith_dataset(name)
    found = False
    examples = LANGSMITH_CLIENT.list_examples(dataset_id=dataset.id)
    for ex in examples:
        if ex.inputs == input and ex.outputs == output:
            found = True
            continue
        LANGSMITH_CLIENT.delete_example(ex.id)
    if not found:
        LANGSMITH_CLIENT.create_example(
            inputs=input,
            outputs=output,
            dataset_id=dataset.id,
        )


@dataclass
class LangSmithFeedback:
    key: str
    score: float
    value: str
    eval_run_id: str


@dataclass
class LangSmithDatasetRunResult:
    run_id: str
    output: Any
    error: Any
    feedbacks: list[LangSmithFeedback]


def run_langchain_chain_on_dataset(
    dataset_name: str, chain_factory: Callable, run_eval_config: RunEvalConfig
) -> List[LangSmithDatasetRunResult]:
    results = LANGSMITH_CLIENT.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=chain_factory,
        evaluation=run_eval_config,
        verbose=True,
    )

    runs = []
    for result in results["results"].values():
        print(result)
        feedbacks = []
        for feedback in result["feedback"]:
            key = feedback.key
            score = feedback.score
            value = feedback.value
            eval_run_id = str(feedback.evaluator_info["__run"].run_id)
            print(
                f"Feedback for {key} is {score} with value {value} for run {eval_run_id}"
            )
            feedbacks.append(LangSmithFeedback(key, score, value, eval_run_id))
        runs.append(
            LangSmithDatasetRunResult(
                result["run_id"],
                result["output"] if "output" in result else None,
                result["Error"] if "Error" in result else None,
                feedbacks,
            )
        )
    return runs


def record_langsmith_sharelink(run_id: str, record_property: Callable) -> None:
    link = get_langsmith_sharelink(run_id=run_id)
    record_property("langsmith_url", link)


def get_langsmith_sharelink(run_id: str, tries: int = 6) -> str:
    ex = None
    while tries > 0:
        try:
            return LANGSMITH_CLIENT.share_run(run_id)
        except Exception as e:
            ex = e
            time.sleep(5)
            tries -= 1
    raise ex

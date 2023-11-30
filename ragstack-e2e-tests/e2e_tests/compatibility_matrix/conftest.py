import json
from typing import List

import pytest
import os

compatibility_matrix_results = {}
TYPE_SIMPLE_RAG = "simple-rag"
TYPE_DOCUMENT_LOADER = "document-loader"


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    info = os.getenv("RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO", "")
    if info:
        info_dict = json.loads(info)
        if "type" not in info_dict:
            raise Exception("Missing type in compatibility matrix info")
        if info_dict["type"] not in compatibility_matrix_results:
            compatibility_matrix_results[info_dict["type"]] = []
        result = {
            "specs": info_dict,
            "result": "✅" if rep.outcome == "passed" else "❌",
            "error": call.excinfo,
        }
        compatibility_matrix_results[info_dict["type"]].append(result)
        os.environ["RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO"] = ""


def set_current_test_info_simple_rag(llm: str, embedding: str, vector_db: str) -> None:
    os.environ["RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO"] = json.dumps(
        {
            "type": TYPE_SIMPLE_RAG,
            "llm": llm,
            "embedding": embedding,
            "vector_db": vector_db,
        }
    )


def set_current_test_info_document_loader(doc_loader: str) -> None:
    os.environ["RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO"] = json.dumps(
        {"type": TYPE_DOCUMENT_LOADER, "doc_loader": doc_loader}
    )


def generate_plain_text_report(tests: dict[str, List[dict]]) -> str:
    report = ""
    if TYPE_SIMPLE_RAG in tests:
        report += "Simple RAG\n"
        report += generate_plain_text_report_simple_rag(tests[TYPE_SIMPLE_RAG])

    if TYPE_DOCUMENT_LOADER in tests:
        report += "Document Loaders\n"
        report += generate_plain_text_report_doc_loader(tests[TYPE_DOCUMENT_LOADER])

    return report


def generate_plain_text_report_simple_rag(tests):
    result = "(Vector, Embedding, LLM)\n\n"
    for test in tests:
        llm = test["specs"]["llm"]
        embedding = test["specs"]["embedding"]
        vector_db = test["specs"]["vector_db"]
        result += "-" * 60 + "\n"
        result += vector_db
        result += " | "
        result += embedding
        result += " | "
        result += llm
        result += ": "
        result += (
            test["result"] + (" " + str(test["error"]) if test["error"] else "") + "\n"
        )
    return result


def generate_plain_text_report_doc_loader(tests):
    result = "(Document Loader)\n\n"
    for test in tests:
        doc_loader = test["specs"]["doc_loader"]
        result += "-" * 60 + "\n"
        result += doc_loader
        result += ": "
        result += (
            test["result"] + (" " + str(test["error"]) if test["error"] else "") + "\n"
        )
    return result


def generate_markdown_report(tests: dict[str, List[dict]]) -> str:
    report = "## Compatibility matrix results\n"
    if TYPE_SIMPLE_RAG in tests:
        report += "\n### Simple RAG\n\n"
        report += generate_markdown_report_simple_rag(tests[TYPE_SIMPLE_RAG])
    if TYPE_DOCUMENT_LOADER in tests:
        report += "\n### Document Loaders\n\n"
        report += generate_markdown_report_doc_loader(tests[TYPE_DOCUMENT_LOADER])
    return report


def generate_markdown_report_simple_rag(tests):
    report = "| Vector DB | Embedding | LLM | Result |\n"
    report += "|-----------|-----------|-----|--------|\n"
    for test in tests:
        llm = test["specs"]["llm"]
        embedding = test["specs"]["embedding"]
        vector_db = test["specs"]["vector_db"]
        result = test["result"] + (" " + str(test["error"]) if test["error"] else "")
        report += f"| {vector_db} | {embedding} | {llm} | {result} |\n"

    return report


def generate_markdown_report_doc_loader(tests):
    report = "| Document Loader | Result |\n"
    report += "|-----------------|--------|\n"
    for test in tests:
        doc_loader = test["specs"]["doc_loader"]
        result = test["result"] + (" " + str(test["error"]) if test["error"] else "")
        report += f"| {doc_loader} | {result} |\n"
    return report


@pytest.fixture(scope="session", autouse=True)
def dump_report():
    yield
    print("Compatibility matrix results:")
    print(compatibility_matrix_results)
    with open("generated-compatibility-matrix.md", "w") as f:
        f.write(generate_markdown_report(compatibility_matrix_results))
    with open("generated-compatibility-matrix.txt", "w") as f:
        f.write(generate_plain_text_report(compatibility_matrix_results))


def get_required_env(name) -> str:
    if name not in os.environ:
        pytest.skip(f"Missing required environment variable: {name}")
    return os.environ[name]

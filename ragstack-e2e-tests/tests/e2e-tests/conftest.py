import json
import unicodedata
from typing import List

import pytest
import os

compatibility_matrix_results = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    info = os.getenv("RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO", "")
    if info:
        info_dict = json.loads(info)
        result = {
            "specs": info_dict,
            "result": "✅" if rep.outcome == "passed" else "❌",
            "error": call.excinfo,
        }
        compatibility_matrix_results.append(result)
        os.environ["RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO"] = ""

def set_current_test_info(llm: str, embedding: str, vector_db: str) -> None:
    os.environ["RAGSTACK_E2E_TESTS_COMPATIBILITY_MATRIX_INFO"] = json.dumps({
        "llm": llm,
        "embedding": embedding,
        "vector_db": vector_db
    })


def generate_plain_text_report(tests: List[dict]) -> str:
    result = ""
    for test in tests:
        llm = test['specs']['llm']
        embedding = test['specs']['embedding']
        vector_db = test['specs']['vector_db']
        result += "Vector: " + vector_db + "\n"
        result += "Embedding: " + embedding + "\n"
        result += "LLM: " + llm + "\n"
        result += "Result: " + test['result'] + (" " + str(test['error']) if test['error'] else "") + "\n"
        result += "-" * 80 + "\n"
    return result

def generate_markdown_report(tests: List[dict]) -> str:
    report = "## Compatibility matrix results\n\n"
    report += "| Vector DB | Embedding | LLM | Result |\n"
    report += "|-----------|-----------|-----|--------|\n"

    for test in tests:
        llm = test['specs']['llm']
        embedding = test['specs']['embedding']
        vector_db = test['specs']['vector_db']
        result = test['result'] + (" " + str(test['error']) if test['error'] else "")
        report += f"| {vector_db} | {embedding} | {llm} | {result} |\n"

    return report


@pytest.fixture(scope='session', autouse=True)
def dump_report():
    yield
    print("Compatibility matrix results:")
    print(compatibility_matrix_results)
    with open("generated-compatibility-matrix.md", "w") as f:
        f.write(generate_markdown_report(compatibility_matrix_results))
    with open("generated-compatibility-matrix.txt", "w") as f:
        f.write(generate_plain_text_report(compatibility_matrix_results))




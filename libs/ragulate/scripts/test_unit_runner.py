import sys

import pytest


def main():
    sys.exit(pytest.main(["tests/unit_tests"]))

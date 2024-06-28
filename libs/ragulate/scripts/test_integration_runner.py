import sys

import pytest


def main():
    sys.exit(pytest.main(["tests/integration_tests"]))

#!/usr/bin/env python
import sys

import pytest


def main():
    sys.exit(pytest.main(["tests/unit_tests"]))


if __name__ == "__main__":
    main()

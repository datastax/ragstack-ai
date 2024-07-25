#!/usr/bin/env python
import sys

import pytest


def main() -> None:
    """Run the integration tests."""
    sys.exit(pytest.main(["tests/integration_tests"]))


if __name__ == "__main__":
    main()

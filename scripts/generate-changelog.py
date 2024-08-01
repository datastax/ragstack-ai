#!/usr/bin/env python
import sys

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)
import sys

IMPORTANT_DEPENDENCIES = [
    "ragstack-ai-langchain",
    "ragstack-ai-colbert",
    "ragstack-ai-llamaindex",
    "langchain",
    "langchain-astradb",
    "langchain-community",
    "langchain-core",
    "llama-index",
    "astrapy",
    "cassio",
    "unstructured",
    "colbert-ai",
    "torch",
    "pyarrow",
]

_MIN_ARGV_LEN = 2


def main() -> None:
    if len(sys.argv) < _MIN_ARGV_LEN:
        print("Usage: generate-changelog.py {version} {package_name}")
        sys.exit(1)
    package_version = sys.argv[1]
    root_package_name = sys.argv[2]
    url = f"https://pypi.org/pypi/{root_package_name}/{package_version}/json"
    deps_str = ""
    json_response = requests.get(url, timeout=30).json()
    requires = json_response["info"]["requires_dist"]
    for req in requires:
        version_range = ""
        extra = ""
        require = req.replace(" ", "")
        if ";extra" in require:
            extra = require[require.index(";extra") + 8 :]
            require = require[: require.index(";extra")]

        for i in range(len(require)):
            if require[i] == "=":
                package_name = require[:i]
                version_range = require[i:]
                break
            if require[i] == ">" or require[i] == "<":
                package_name = require[:i]
                version_range = require[i:]
                break
        if not version_range:
            raise ValueError(f"Could not parse version range from {require}")
        for important_dependency in IMPORTANT_DEPENDENCIES:
            if package_name.startswith(important_dependency + "["):
                package_name = important_dependency
                break
        if package_name in IMPORTANT_DEPENDENCIES:
            if package_name == "langchain":
                version_range = (
                    "https://datastax.github.io/ragstack-ai/api_reference/"
                    f"{package_version}/langchain[{version_range}]"
                    "{external-link-icon}"
                )
            extra_str = f" (via extra `{extra}`)" if extra else ""
            deps_str += f"\n| {package_name}{extra_str}\n| {version_range}\n"

    release_date = json_response["urls"][0]["upload_time"][:10]

    print(f"""
== `{root_package_name}`@{package_version} ({release_date})

[caption=]
.Requirements
[%autowidth]
[cols="2*",options="header"]
|===
| Library | Version

{deps_str}

|===
    """)


if __name__ == "__main__":
    main()

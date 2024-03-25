try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)
import sys

IMPORTANT_DEPENDENCIES = ["langchain", "langchain-astradb", "langchain-community", "langchain-core", "llama-index", "astrapy", "cassio", "unstructured"]


def main():
    if len(sys.argv) < 2:
        print("Usage: generate-changelog.py {version}")
        sys.exit(1)
    package_version = sys.argv[1]
    url = f"https://pypi.org/pypi/ragstack-ai/{package_version}/json"
    deps_str = ""
    json_response = requests.get(url).json()
    requires = json_response['info']['requires_dist']
    for require in requires:
        split = require.split(' ')
        package_name = split[0]
        for important_dependency in IMPORTANT_DEPENDENCIES:
            if package_name.startswith(important_dependency + "["):
                package_name = important_dependency
                break
        if package_name in IMPORTANT_DEPENDENCIES:
            version_range = split[1].replace('(', '').replace(')', '')
            if package_name == "langchain":
                version_range = f"https://datastax.github.io/ragstack-ai/api_reference/{package_version}/langchain[{version_range}]{{external-link-icon}}" # noqa

            deps_str += f"\n| {package_name}\n| {version_range}\n"

    release_date = json_response["urls"][0]["upload_time"][:10]

    print(f"""
== {package_version} ({release_date})

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

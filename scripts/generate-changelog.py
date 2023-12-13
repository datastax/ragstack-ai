import sys

import requests

try:
    import toml
except ImportError:
    print("Please install toml package")
    sys.exit(1)

IMPORTANT_DEPENDENCIES = ["langchain", "llama-index", "astrapy", "cassio"]


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

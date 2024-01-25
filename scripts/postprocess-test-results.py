import os.path
import sys
import xml.etree.ElementTree as ET


def rewrite_name(name):
    params = ""
    if "[" in name:
        after_split = name.split("[")
        name = after_split[0]
        params = after_split[1]
        params = params.replace("]", "")
        params = params.replace("_", " ").replace("-", " | ")
        params = f": {params}"

    name = name.replace("_", " ").capitalize()

    return f"{name}{params}"


def main(input_file: str, output_file: str):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find the element you want to modify and change its attribute or text
    for test_suites in root.iter("testsuites"):  # replace 'element_name' with the name of the element you want to find
        for test_suite in test_suites.iter("testsuite"):
            test_suite.set("hostname", "RAGStack CI")
            for test_case in test_suite.iter("testcase"):
                if test_case.find("skipped") is not None:
                    test_suite.remove(test_case)
                test_case.set("name", rewrite_name(test_case.get("name")))

    print(f"Writing modified file to {output_file}")
    tree.write(output_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: postprocess-test-results.py {junit-report-file.xml} {output-file.xml}")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

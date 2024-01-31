import os.path
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List


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


@dataclass
class Link:
    name: str
    url: str


@dataclass
class TestCase:
    name: str
    passed: bool
    time: str
    failure_error_title: str
    failure_error_message: str
    links: List[Link]


@dataclass
class TestSuite:
    name: str
    test_cases: List[TestCase]


def unsafe_escape_data(text):
    if text is None:
        return ""
    if text.startswith("<![CDATA[") and text.endswith("]]>"):
        return text
    if "&" in text:
        text = text.replace("&", "&amp;")
    if "<" in text:
        text = text.replace("<", "&lt;")
    if ">" in text:
        text = text.replace(">", "&gt;")
    return text


def cdata(text: str) -> str:
    return f"<![CDATA[{text}]]>"


import xml

xml.etree.ElementTree._escape_cdata = unsafe_escape_data


def main(input_file: str, output_file: str):
    test_suites = parse_report(input_file)
    if len(test_suites) == 0:
        print("No test suites found in input file, exiting")
    else:
        reporter = generate_new_report(test_suites)
        print(f"Writing report file to {output_file}")
        root = ET.ElementTree(reporter)
        root.write(output_file, encoding="utf-8")


def generate_new_report(test_suites):
    reporter = ET.Element("reporter")
    for test_suite in test_suites.values():
        test_suite_el = ET.Element("test_suite")
        test_suite_el.set("name", test_suite.name)
        for test_case in test_suite.test_cases:
            print(test_case)
            test_case_el = ET.Element("test_case")
            test_case_el.set("name", test_case.name)
            test_case_el.set("status", "passed" if test_case.passed else "failed")
            test_case_el.set("duration", test_case.time)
            if not test_case.passed:
                failure_el = ET.Element("annotation")
                failure_el.set("name", "Failure")
                failure_el.set("level", "error")
                title = ET.SubElement(failure_el, "comment")
                title.set("label", "Error")
                title.text = cdata(test_case.failure_error_title)
                if test_case.failure_error_message:
                    stack = ET.SubElement(failure_el, "comment")
                    stack.set("label", "Stacktrace")
                    stack.text = cdata(test_case.failure_error_message)

                test_case_el.append(failure_el)
            for link in test_case.links:
                link_el = generate_link_annotation(link)
                test_case_el.append(link_el)
            test_suite_el.append(test_case_el)
        reporter.append(test_suite_el)
    github_url = os.environ.get("TESTSPACE_REPORT_GITHUB_URL")
    if github_url:
        link = Link(name="See logs on Github Actions", url=github_url)
        link_el = generate_link_annotation(link)
        reporter.append(link_el)
    return reporter


def generate_link_annotation(link):
    link_el = ET.Element("annotation")
    link_el.set("name", link.name)
    link_el.set("level", "info")
    link_el.set("file", link.url)
    link_el.set("link_file", "true")
    return link_el


def parse_report(input_file: str):
    tree = ET.parse(input_file)
    root = tree.getroot()
    report_test_suites = {}
    for test_suites in root.iter("testsuites"):
        for test_suite in test_suites.iter("testsuite"):
            test_suite.set("hostname", "RAGStack CI")
            for test_case in test_suite.iter("testcase"):
                print("processing test case: " + str(test_case.attrib))
                classname = test_case.get("classname")
                if test_case.find("skipped") is not None or not test_case.get("name") or not classname:
                    continue

                failure = test_case.find("failure")
                if failure is not None:
                    passed = False
                    failure_error_title = failure.get("message")
                    failure_error_message = failure.text
                else:
                    passed = True
                    failure_error_title = ""
                    failure_error_message = ""

                properties = test_case.find("properties")
                links = []
                if properties:
                    for prop in properties.iter("property"):
                        if prop.get("name") == "langsmith_url":
                            links.append(Link(name="LangSmith trace", url=prop.get("value")))

                report_test_case = TestCase(
                    name=rewrite_name(test_case.get("name")),
                    passed=passed,
                    time=str(float(test_case.get("time")) * 1000),
                    failure_error_title=failure_error_title,
                    failure_error_message=failure_error_message,
                    links=links,
                )
                if classname not in report_test_suites:
                    report_test_suites[classname] = TestSuite(
                        name=classname, test_cases=[]
                    )
                report_test_suites[classname].test_cases.append(report_test_case)
    return report_test_suites


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: generate-testspace-report.py {junit-report-file.xml} {output-file.xml}")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

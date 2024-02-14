import os.path
import sys
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List


def rewrite_test_case_name(name):
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

def beatify_xml_name(name):
    return name.replace("_", " ").capitalize()


@dataclass
class Link:
    name: str
    description: str
    url: str
    level: str


@dataclass
class Failure:
    title: str
    description: str


@dataclass
class TestCase:
    name: str
    passed: bool
    time: str
    links: List[Link]
    failures: List[Failure]


@dataclass
class TestSuite:
    name: str
    test_cases: List[TestCase]
    links: List[Link]


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


def main(type: str, input_file: str, output_file: str):
    if type == "tests":
        test_suites = parse_test_report(input_file)
        if len(test_suites) == 0:
            print("No test suites found in input file, exiting")
            return
    else:
        test_suites = parse_snyk_report(input_file)

    reporter = generate_new_report(test_suites)
    print(f"Writing report file to {output_file}")
    root = ET.ElementTree(reporter)
    root.write(output_file, encoding="utf-8")


def generate_new_report(test_suites):
    reporter = ET.Element("reporter")
    for test_suite in test_suites.values():
        test_suite_el = ET.Element("test_suite")
        test_suite_el.set("name", test_suite.name)
        for link in test_suite.links:
            link_el = generate_link_annotation(link)
            test_suite_el.append(link_el)
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
                for failure in test_case.failures:
                    title = ET.SubElement(failure_el, "comment")
                    title.set("label", "Error")
                    title.text = cdata(failure.title)
                    if failure.description:
                        stack = ET.SubElement(failure_el, "comment")
                        stack.set("label", "Stacktrace")
                        stack.text = cdata(failure.description)

                    test_case_el.append(failure_el)
            for link in test_case.links:
                link_el = generate_link_annotation(link)
                test_case_el.append(link_el)
            test_suite_el.append(test_case_el)
        reporter.append(test_suite_el)
    github_url = os.environ.get("TESTSPACE_REPORT_GITHUB_URL")
    if github_url:
        link = Link(name="See logs on Github Actions", url=github_url, description="", level="info")
        link_el = generate_link_annotation(link)
        reporter.append(link_el)
    return reporter


def generate_link_annotation(link):
    link_el = ET.Element("annotation")
    link_el.set("name", link.name)
    if link.description:
        link_el.set("description", link.description)
    link_el.set("level", link.level or "info")
    if link.url:
        link_el.set("file", link.url)
        link_el.set("link_file", "true")
    return link_el


SNYK_REPORT_SUITE_NAME = "Security scans"
def parse_snyk_report(input_file: str):
    test_cases = []
    vulnerabilities = {}
    files = []

    if os.path.isdir(input_file):
        for f in os.listdir(input_file):
            if f.endswith(".json"):
                files.append(f)
    else:
        files.append(input_file)
    all_links = []

    for snykfile in files:
        snykfile = os.path.join(input_file, snykfile)
        print("Reading file: " + snykfile)

        with open(snykfile, "r") as file:
            data = json.load(file)
            for vulnerability in data.get("vulnerabilities", []):
                title = vulnerability.get("title", "?")
                cvssScore = vulnerability.get("cvssScore", "")
                severity = vulnerability.get("severity", "")
                from_packages = " -> ".join(vulnerability.get("from", []))
                version = vulnerability.get("version", "?")
                id = vulnerability.get("id", "?")
                identifiers = vulnerability.get("identifiers", [])

                if "GHSA" in identifiers:
                    link = f"https://github.com/advisories/{identifiers['GHSA'][0]}"
                elif "CVE" in identifiers:
                    link = f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={identifiers['CVE'][0]}"
                else:
                    link = ""

                ann_title = f"{title}@{version}"
                cvss_str = f" [CVSS: {cvssScore}]" if cvssScore else ""
                ann_description = f"{title} [{id}] [{severity.capitalize()} severity] {cvss_str} from {from_packages}"
                if id not in vulnerabilities:
                    vulnerabilities[id] = [ann_title, ann_description, link]

            project = data.get("projectName", "")
            local_links = []
            if "docker-image" in project:
                test_case_name = "Docker image"
                docker_image_digest = os.environ.get("TESTSPACE_REPORT_DOCKER_IMAGE_DIGEST", "")
                if docker_image_digest:
                    link = Link(name="Docker image", description=docker_image_digest, url="", level="info")
                    local_links.append(link)
                    all_links.append(link)
            else:
                test_case_name = "Python dependencies"

            passed = len(vulnerabilities) == 0
            for v in vulnerabilities.values():
                link = Link(name=v[0], description=v[1], url=v[2], level="error")
                all_links.append(link)
                local_links.append(link)
            test_case = TestCase(name=test_case_name, passed=passed, time="0.0", failures=[], links=local_links)
            test_cases.append(test_case)

    return {SNYK_REPORT_SUITE_NAME: TestSuite(name=SNYK_REPORT_SUITE_NAME, test_cases=test_cases, links=all_links)}


def parse_test_report(input_file: str):
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
                failures = []
                if failure is not None:
                    failures.append(Failure(title=failure.get("message"), description=failure.text))

                properties = test_case.find("properties")
                links = []
                if properties:
                    prop_map = {}
                    for prop in properties.iter("property"):
                        prop_map[prop.get("name")] = prop.get("value")

                    for prop in properties.iter("property"):
                        name = prop.get("name")
                        if name == "langsmith_url":
                            links.append(
                                Link(name="LangSmith trace", url=prop.get("value"), level="info", description=""))
                        elif name.startswith("langsmith_feedback_"):
                            if not name.endswith("url"):
                                url_name = name + "_url"
                                url_value = ""
                                if url_name in prop_map:
                                    url_value = prop_map[url_name]
                                fname = beatify_xml_name(name.replace("langsmith_feedback_", ""))
                                desc = f"{fname}: {prop.get('value')}"
                                links.append(
                                    Link(name="LangSmith feedback", url=url_value, level="info", description=desc))

                report_test_case = TestCase(
                    name=rewrite_test_case_name(test_case.get("name")),
                    passed=len(failures) == 0,
                    time=str(float(test_case.get("time")) * 1000),
                    failures=failures,
                    links=links,
                )
                if classname not in report_test_suites:
                    report_test_suites[classname] = TestSuite(
                        name=classname, test_cases=[], links=[]
                    )
                report_test_suites[classname].test_cases.append(report_test_case)
    return report_test_suites


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: generate-testspace-report.py [tests|snyk] {junit-file.xml|snyk.json} {output-file.xml}")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])

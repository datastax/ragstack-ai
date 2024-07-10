#!/usr/bin/env python

import json
import sys

_MIN_ARGV_LEN = 3

if len(sys.argv) < _MIN_ARGV_LEN:
    print("Usage: parse-snyk-report.py {input_json} {output_file}")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file) as file:
    data = json.load(file)
    vulnerabilities = data.get("vulnerabilities", [])

    with open(output_file, "w") as output:
        for vulnerability in vulnerabilities:
            title = vulnerability.get("title", "N/A")
            cvss_score = vulnerability.get("cvssScore", "N/A")
            from_packages = " -> ".join(vulnerability.get("from", []))

            report_line = f"{title} | {from_packages} | {cvss_score}\n"
            output.write(report_line)

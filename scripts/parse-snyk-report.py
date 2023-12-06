import json
import sys

if len(sys.argv) < 3:
    print("Usage: parse-snyk-report.py {input_json} {output_file}")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as file:
    data = json.load(file)
    vulnerabilities = data.get('vulnerabilities', [])

    with open(output_file, 'w') as output:
        for vulnerability in vulnerabilities:
            title = vulnerability.get('title', 'N/A')
            cvssScore = vulnerability.get('cvssScore', 'N/A')
            from_packages = ', '.join(vulnerability.get('from', []))

            report_line = f"{title} - {from_packages} - {cvssScore}\n"
            output.write(report_line)

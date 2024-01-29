import argparse
import base64
import json
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import os

PERCENTILES = [50, 90, 95, 99]


def extract_values_from_result_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    all_values = [run["values"][0] for run in data["benchmarks"][0]["runs"]]
    percentiles_values = {}
    for p in PERCENTILES:
        percentiles_values["p" + str(p)] = round(np.percentile(all_values, p), 2)
    return {
        "values": percentiles_values,
        "name": os.path.basename(file_path).replace(".json", ""),
    }


def scan_result_directory(directory_path, filter_by):
    values = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json") and filter_by in file_name:
            file_path = os.path.join(directory_path, file_name)
            values.append(extract_values_from_result_file(file_path))
    return sorted(values, key=lambda v: v["values"]["p99"])


def render_plot(values, export_to: str, show: bool = False):
    plt = render_plot_obj(values)
    plt.savefig(export_to)
    if show:
        plt.show()


def render_plot_obj(values):
    plt.figure(figsize=(10, 8))
    cols = []
    rows = []
    cells = []
    for p in PERCENTILES:
        cols.append(f"p{p}")
    for i, value in enumerate(values):
        plt.plot(value["values"].keys(), value["values"].values(), label=value["name"])
        cells.append(list(value["values"].values()))
        rows.append(value["name"])
    plt.title("All")
    plt.xlabel("Percentile")
    plt.ylabel("Milliseconds")
    plt.legend(bbox_to_anchor=(0, -0.2), loc="upper left", ncol=1)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.5)
    return plt


def render_table(values):
    cols = ["Test Case"]

    from prettytable import PrettyTable

    x = PrettyTable()
    for p in PERCENTILES:
        cols.append(f"p{p}")

    x.field_names = cols

    for i, value in enumerate(values):
        x.add_row([value["name"]] + list(value["values"].values()))
    print(x)


def render_html_table(headers, rows):
    table = """<table class="w-full divide-y divide-gray-200 border border-collapse">"""
    table += """<thead class="bg-gray-100 text-xs font-semibold uppercase text-center text-gray-800"><tr>"""
    for header in headers:
        table += f"<th class='px-6 py-3'>{header}</th>"
    table += "</tr></thead>"
    table += """<tbody class="bg-white divide-y divide-gray-200 text-center text-sm">"""
    for row in rows:
        table += """<tr class="hover:bg-gray-100">"""
        for cell in row:
            table += f"<td class='px-6 py-4'>{cell}</td>"
        table += "</tr>"
    table += "</tbody></table>"
    return table


def render_html(values, export_to: str):
    title = "RAGStack - Benchmarks Report"
    import datetime

    title += " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = []
    for i, value in enumerate(values):
        ps = []
        for p in PERCENTILES:
            k = f"p{p}"
            ps.append(value["values"][k])
        table_rows.append([value["name"]] + ps)
    table = render_html_table(
        ["Test Case"] + [f"p{p}" for p in PERCENTILES], table_rows
    )

    plot_obj = render_plot_obj(values)
    tmpfile = BytesIO()
    plot_obj.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    plot = f"<img src='data:image/png;base64,{encoded}'>"

    html = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body>
            <header class="bg-gray-100 py-6 px-4 text-center">
                <h1 class="text-3xl font-bold text-gray-800">{title}</h1>
            </header>
             <div class="container mx-auto p-4">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div>
                    <h2 class="text-xl font-semibold mb-2">Table</h2>
                    {table}
                  </div>
                  <div>
                    <h2 class="text-xl font-semibold mb-2">Plot</h2>
                    {plot}
                  </div>
                </div>
              </div>
        """

    with open(export_to, "w") as f:
        f.write(html)


def render_markdown(values, export_to: str, plot_src: str = "plot.png"):
    title = "RAGStack - Benchmarks Report"
    import datetime

    title += " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = []
    for i, value in enumerate(values):
        ps = []
        for p in PERCENTILES:
            k = f"p{p}"
            ps.append(value["values"][k])
        table_rows.append([value["name"]] + ps)

    headers = ["Test Case"] + [f"p{p}" for p in PERCENTILES]

    def draw_row(list):
        return f"| {' | '.join(str(col) for col in list)} |"

    table = draw_row(headers)
    table += "\n|" + "|".join(["---" for _ in headers]) + "|"
    for row in table_rows:
        table += "\n" + draw_row(row)

    plot = f"<img src='{plot_src}' />"

    md = f"""# {title}

{table}

{plot}"""

    with open(export_to, "w") as f:
        f.write(md)


def draw_report(directory_path: str, format: str, filter_by: str):
    values = scan_result_directory(directory_path, filter_by)
    is_all = format == "all"
    if is_all or format == "plot" or format == "plot_svg":
        render_plot(
            values,
            export_to=os.path.join(directory_path, "plot.svg"),
            show=format == "plot",
        )
    if is_all or format == "html":
        render_html(values, export_to=os.path.join(directory_path, "report.html"))
    if is_all or format == "markdown":
        render_plot(
            values,
            export_to=os.path.join(directory_path, "plot.svg"),
            show=False,
        )
        render_markdown(
            values,
            export_to=os.path.join(directory_path, "report.md"),
            plot_src="plot.svg",
        )
    if is_all or format == "table":
        render_table(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmarks runner result visualizer",
        description="Visualize benchmarks results",
    )

    parser.add_argument(
        "-r",
        "--reports-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Reports dir",
    )
    parser.add_argument(
        "--format",
        choices=["table", "plot", "plot_svg", "html", "markdown", "all"],
        default="table",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default="",
        help="Filter results. e.g. to filter only openai, use: openai_",
    )
    args = parser.parse_args()
    draw_report(args.reports_dir, args.format, args.filter)

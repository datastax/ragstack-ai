#!/usr/bin/env python3

import argparse
import base64
import json
import pandas as pd
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import os

PERCENTILES = [50, 90, 95, 99]

# Manually modify these to the desired values to avoid having to input them
GPUS = 0
NUM_CHUNKS = 0
TITLE = None
TITLE = "NeMo Embedding + Astra Indexing: Throughput and Inference Latency (V100)"


def extract_values_from_result_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Get all values from all runs, then flatten them out to a single list
    all_values = [run["values"] for run in data["benchmarks"][0]["runs"]]
    all_values = [v for sublist in all_values for v in sublist]
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


def extract_threads(name):
    """
    Extracts the number of threads from the name, e.g.
    embeddings_batch100_chunk512-nemo_microservice-8
    """
    return int(name.split("-")[-1])


def render_throughput_plots(values):
    """
    Renders throughput and latency plots.

    Note: Only nemo and openai items are supported
    """
    nemo_items = [item for item in values if "nemo" in item["name"]]
    openai_items = [item for item in values if "openai" in item["name"]]

    if len(nemo_items) > 0:
        nemo_items_sorted = sorted(nemo_items, key=lambda x: extract_threads(x["name"]))
        _render_throughput_plot(nemo_items_sorted, "NeMo")

    if len(openai_items) > 0:
        openai_items_sorted = sorted(
            openai_items, key=lambda x: extract_threads(x["name"])
        )
        _render_throughput_plot(openai_items_sorted, "OpenAI")


def _render_throughput_plot(sorted_items, name):
    print(f"Rendering plot for {name}")
    threads = [extract_threads(item["name"]) for item in sorted_items]
    # Ideally, this is the chunk size, but have to figure out why setting
    # chunk size to 512 is going over token limit
    # chunk_sizes = [
    #     int(item["name"].split("_")[2].split("-")[0].split("chunk")[1])
    #     for item in sorted_items
    # ]
    # if len(set(chunk_sizes)) > 1:
    #     raise ValueError("Throughput plots only work with a single chunk size")
    chunk_size = input("Chunk size? ")
    chunk_sizes = [int(chunk_size) for _ in range(len(threads))]

    batch_sizes = [
        int(item["name"].split("_")[1].split("batch")[1]) for item in sorted_items
    ]
    if len(set(batch_sizes)) > 1:
        raise ValueError("Throughput plots only work with a single batch size")

    if GPUS == 0:
        gpu = input("How many GPUs were used? ")
        gpu = int(gpu)
        gpus = [gpu for _ in range(len(threads))]
    if NUM_CHUNKS == 0:
        chunk = input("How many chunks were created (check benchmarks.log)? ")
        chunk = int(chunk)
        chunks = [chunk for _ in range(len(threads))]

    p99_values = [item["values"]["p99"] for item in sorted_items]
    throughput = [chunk / p for p in p99_values]
    avg_inference = [1000 / t for t in throughput]

    data = {
        "Request Concurrency": threads,
        "GPUs (V100)": gpus,
        "Chunks": chunks,
        "Approx. Chunk Size (B)": chunk_sizes,
        "Benchmark Batch Size": batch_sizes,
        "Throughput (infer/sec)": throughput,
        "p99 Latency / Benchmark (s)": p99_values,
        "Avg. Latency / Inference (ms)": avg_inference,
    }
    df = pd.DataFrame(data)

    print(f"Saving markdown for {name} to benchmark_results-{name}.md")
    markdown_string = df.to_markdown(index=False)
    with open(f"benchmark_results-{name}.md", "w") as f:
        f.write(markdown_string)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = "tab:red"
    ax1.set_xlabel("Request Concurrency")
    ax1.set_ylabel("Throughput (infer/sec)", color=color)
    ax1.plot(
        df["Request Concurrency"],
        df["Throughput (infer/sec)"],
        color=color,
        marker="o",
        label="Throughput",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Annotating each point with its value on ax1
    for x, y in zip(df["Request Concurrency"], df["Throughput (infer/sec)"]):
        ax1.text(x, y, f"{y:.2f}", color=color, ha="center", va="bottom")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("Avg. Latency / Inference (ms)", color=color)
    ax2.plot(
        df["Request Concurrency"],
        df["Avg. Latency / Inference (ms)"],
        color=color,
        marker="o",
        fillstyle="none",
        label="Latency",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Annotating each point with its value on ax2
    for x, y in zip(df["Request Concurrency"], df["Avg. Latency / Inference (ms)"]):
        ax2.text(x, y, f"{y:.2f}", color=color, ha="center", va="bottom")

    plt.subplots_adjust(top=0.92)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.grid(True)

    title = input("Enter Plot Title: ") if TITLE is None else TITLE
    plt.title(title, pad=5)
    file_name = f"benchmark_results-{name}.png"
    print(f"Saving plot for {name} to {file_name}")
    plt.savefig(file_name, bbox_inches="tight")


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
    plt.ylabel("Seconds")
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


def draw_report(directory_path: str, format: str, filter_by: str, throughput: bool):
    values = scan_result_directory(directory_path, filter_by)
    is_all = format == "all"
    if is_all or format == "plot" or format == "plot_svg":
        if throughput:
            render_throughput_plots(values)
        else:
            render_plot(
                values,
                export_to=os.path.join(directory_path, "plot.svg"),
                show=format == "plot",
            )
    if is_all or format == "html":
        if throughput:
            raise NotImplementedError("Throughput mode is not supported for HTML")
        render_html(values, export_to=os.path.join(directory_path, "index.html"))
    if is_all or format == "markdown":
        render_plot(
            values,
            export_to=os.path.join(directory_path, "plot.svg"),
            show=False,
        )
        render_markdown(
            values,
            export_to=os.path.join(directory_path, "README.md"),
            plot_src="plot.svg",
        )
    if is_all or format == "table":
        if throughput:
            raise NotImplementedError("Throughput mode is not supported for table")
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

    parser.add_argument(
        "-t",
        "--throughput",
        action="store_true",
        help="If this flag is set, more parameters to the markdown and plots will be required and added. Only `markdown` and `plot` is supported",
    )

    args = parser.parse_args()
    draw_report(args.reports_dir, args.format, args.filter, args.throughput)

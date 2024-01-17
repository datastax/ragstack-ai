import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

PERCENTILES = [50, 90, 95, 99]


def extract_values_from_result_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    all_values = [run['values'][0] for run in data['benchmarks'][0]['runs']]
    percentiles_values = {}
    for p in PERCENTILES:
        percentiles_values["p" + str(p)] = round(np.percentile(all_values, p), 2)
    return {"values": percentiles_values, "name": os.path.basename(file_path).replace(".json", "")}


def scan_result_directory(directory_path, filter_by):
    values = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json') and filter_by in file_name:
            file_path = os.path.join(directory_path, file_name)
            values.append(extract_values_from_result_file(file_path))
    return values


def render_plot(values):
    plt.figure(figsize=(10, 8))
    cols = []
    rows = []
    cells = []
    for p in PERCENTILES:
        cols.append(f'p{p}')

    for i, value in enumerate(values):
        plt.plot(value["values"].keys(), value["values"].values(), label=value["name"])
        cells.append(list(value["values"].values()))
        rows.append(value["name"])

    plt.title(f'All')

    plt.table(cellText=cells,
              rowLabels=rows,
              colLabels=cols,
              loc='bottom')

    plt.xlabel('Percentile')
    plt.ylabel('Milliseconds')
    plt.legend(bbox_to_anchor=(0, -0.2), loc='upper left', ncol=1)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


def render_table(values):
    values = sorted(values, key=lambda v: v["values"]["p99"])
    cols = ["Model"]

    from prettytable import PrettyTable
    x = PrettyTable()
    for p in PERCENTILES:
        cols.append(f'p{p}')

    x.field_names = cols

    for i, value in enumerate(values):
        x.add_row([value["name"]] + list(value["values"].values()))
    print(x)


def draw_report(directory_path: str, format: str, filter_by: str):
    values = scan_result_directory(directory_path, filter_by)
    if format == "plot":
        render_plot(values)
    else:
        render_table(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Benchmarks runner result visualizer',
        description='Visualize benchmarks results')

    parser.add_argument('-r', '--reports-dir', type=str, default=os.path.join(os.path.dirname(__file__), "reports"),
                        help='Reports dir')
    parser.add_argument('--format', choices=["table", "plot"], default="table")
    parser.add_argument('-f', '--filter', type=str, default="",
                        help="Filter results. e.g. to filter only openai, use: openai_")
    args = parser.parse_args()
    draw_report(args.reports_dir, args.format, args.filter)

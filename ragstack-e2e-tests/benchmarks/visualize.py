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
        percentiles_values["p" + str(p)] = np.percentile(all_values, p)
    return {"values": percentiles_values, "name": os.path.basename(file_path).replace(".json", "")}


def scan_result_directory(directory_path):
    values = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json') and "" in file_name:
            file_path = os.path.join(directory_path, file_name)
            values.append(extract_values_from_result_file(file_path))
    return values


def draw_report():
    values = scan_result_directory("reports")

    plt.figure(figsize=(10, 8))
    for i, value in enumerate(values):
        plt.plot(value["values"].keys(), value["values"].values(), label=value["name"])
    plt.title(f'All')
    plt.xlabel('Percentile')
    plt.ylabel('Milliseconds')
    plt.legend(bbox_to_anchor=(0, -0.2), loc='upper left', ncol=1)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.5)
    plt.show()



if __name__ == '__main__':
    draw_report()

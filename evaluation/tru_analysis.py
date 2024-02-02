import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _gen_visibility_by_metric(test_count: int, metric_count: int, metric_index: int):
    visibility = []
    for m in range(metric_count):
        for t in range(test_count):
            visibility.append(m == metric_index)
    return visibility


def box_plot_by_metric(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)
    tests = df["test"].unique().tolist()
    datasets = df["dataset"].unique().tolist()
    metrics = ["groundedness", "context_relevance",
               "answer_correctness", "answer_relevance", "latency"]

    # generate an array of rainbow colors by fixing the saturation and lightness of the HSL
    # representation of color and marching around the hue.
    c = ["hsl("+str(h)+",50%"+",50%)" for h in np.linspace(0, 360, len(tests) + 1)]

    fig = go.Figure()
    metric_dropdown_items = []

    metric_index = 0
    for metric in metrics:
        visible = metric == metrics[0]
        metric_dropdown_items.append({
            "label": metric,
            "method": "update",
            "args": [
                {"visible": _gen_visibility_by_metric(
                    len(tests), len(metrics), metric_index)}
            ]
        })
        test_index = 0
        for test in tests:
            fig.add_trace(go.Box(
                y=df["dataset"][df["test"] == test],
                x=df[metric][df["test"] == test],
                name=test,
                marker_color=c[test_index],
                visible=visible,
            ))
            test_index += 1
        metric_index += 1

    height = (len(datasets) * len(tests) * 20) + 150

    fig.update_traces(orientation="h", boxmean=True, jitter=1, )
    fig.update_layout(boxmode="group", height=height, width=900)
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(yaxis_title="dataset", xaxis_title="score")

    metric_dropdown = dict(active=0, buttons=metric_dropdown_items,
                           yanchor="bottom", y=1.05, xanchor="right", x=-.1)

    range_dropdown_items = [
        {"label": "All Values", "method": "relayout",
            "args": [{"xaxis": {"rangemode": "normal"}}]},
        {"label": "Non Negative", "method": "relayout",
            "args": [{"xaxis": {"rangemode": "nonnegative"}}]},
    ]

    range_dropdown = dict(active=0, buttons=range_dropdown_items,
                          yanchor="bottom", y=1.01, xanchor="right", x=-.1)

    fig.update_layout(updatemenus=[metric_dropdown, range_dropdown])
    fig.show()


def _gen_visibility_by_dataset(test_count: int, dataset_count: int, metric_index: int):
    visibility = []
    for m in range(dataset_count):
        for t in range(test_count):
            visibility.append(m == metric_index)
    return visibility


def box_plot_by_dataset(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)
    tests = df["test"].unique().tolist()
    datasets = df["dataset"].unique().tolist()
    metrics = ["groundedness", "context_relevance",
               "answer_relevance", "answer_correctness",]

    # generate an array of rainbow colors by fixing the saturation and lightness of the HSL
    # representation of color and marching around the hue.
    c = ["hsl("+str(h)+",50%"+",50%)" for h in np.linspace(0, 360, len(tests) + 1)]

    fig = go.Figure()
    dataset_dropdown_items = []

    dataset_index = 0
    for dataset in datasets:
        visible = dataset == datasets[0]
        dataset_dropdown_items.append({
            "label": dataset,
            "method": "update",
            "args": [
                {"visible": _gen_visibility_by_dataset(
                    len(tests), len(datasets), dataset_index)}
            ]
        })
        test_index = 0
        for test in tests:
            y = []
            x = []
            for metric in metrics:
                dx = df[metric][df["test"] == test][df["dataset"] == dataset]
                x.extend(dx)
                y.extend([metric] * len(dx))

            fig.add_trace(go.Box(
                y=y,
                x=x,
                name=test,
                marker_color=c[test_index],
                visible=visible,
            ))
            test_index += 1
        dataset_index += 1

    height = (len(metrics) * len(tests) * 20) + 150

    fig.update_traces(orientation="h", boxmean=True, jitter=1, )
    fig.update_layout(boxmode="group", height=height, width=900)
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(yaxis_title="metric", xaxis_title="score")

    dataset_dropdown = dict(active=0, buttons=dataset_dropdown_items,
                            yanchor="bottom", y=1.09, xanchor="right", x=0)

    range_dropdown_items = [
        {"label": "All Values", "method": "relayout",
            "args": [{"xaxis": {"rangemode": "normal"}}]},
        {"label": "Non Negative", "method": "relayout",
            "args": [{"xaxis": {"rangemode": "nonnegative"}}]},
    ]

    range_dropdown = dict(active=0, buttons=range_dropdown_items,
                          yanchor="bottom", y=1.01, xanchor="right", x=-.1)

    fig.update_layout(updatemenus=[dataset_dropdown, range_dropdown])
    fig.show()

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from pandas import DataFrame
from plotly.io import write_image

from .utils import get_tru


class Analysis:

    def get_all_data(self, recipes: List[str]) -> DataFrame:
        df_all = pd.DataFrame()

        all_metrics: List[str] = []

        for recipe in recipes:
            tru = get_tru(recipe_name=recipe)

            for app in tru.get_apps():
                dataset = app["app_id"]
                df, metrics = tru.get_records_and_feedback([dataset])
                all_metrics.extend(metrics)

                columns_to_keep = metrics + [
                    "record_id",
                    "latency",
                    "total_tokens",
                    "total_cost",
                ]
                columns_to_drop = [
                    col for col in df.columns if col not in columns_to_keep
                ]

                df.drop(columns=columns_to_drop, inplace=True)
                df["recipe"] = recipe
                df["dataset"] = dataset

                # set negative values to None
                for metric in metrics:
                    df.loc[df[metric] < 0, metric] = None

                df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

            tru.delete_singleton()

        df_all.reset_index(drop=True, inplace=True)

        return df_all, list(set(all_metrics))

    def calculate_statistics(self, df: pd.DataFrame, metrics: list):
        stats = {}
        for recipe in df["recipe"].unique():
            stats[recipe] = {}
            for metric in metrics:
                stats[recipe][metric] = {}
                for dataset in df["dataset"].unique():
                    data = df[(df["recipe"] == recipe) & (df["dataset"] == dataset)][
                        metric
                    ]
                    stats[recipe][metric][dataset] = {
                        "high": data.max(),
                        "low": data.min(),
                        "median": data.median(),
                        "mean": data.mean(),
                        "1st_quartile": data.quantile(0.25),
                        "3rd_quartile": data.quantile(0.75),
                    }
        return stats

    def output_box_plots_by_dataset(self, df: DataFrame, metrics: List[str]):
        stats = self.calculate_statistics(df, metrics)
        recipes = sorted(df["recipe"].unique(), key=lambda x: x.lower())
        datasets = sorted(df["dataset"].unique(), key=lambda x: x.lower())
        metrics = sorted(metrics)
        metrics.reverse()

        # generate an array of rainbow colors by fixing the saturation and lightness of the HSL
        # representation of color and marching around the hue.
        c = [
            "hsl(" + str(h) + ",50%" + ",50%)"
            for h in np.linspace(0, 360, len(recipes) + 1)
        ]

        height = max((len(metrics) * len(recipes) * 20) + 150, 450)

        for dataset in datasets:
            fig = go.Figure()
            test_index = 0
            for recipe in recipes:
                y = []
                x = []
                q1 = []
                median = []
                q3 = []
                mean = []
                low = []
                high = []
                for metric in metrics:
                    stat = stats[recipe][metric][dataset]
                    y.append(metric)
                    x.append(stat["mean"])
                    q1.append(stat["1st_quartile"])
                    median.append(stat["median"])
                    q3.append(stat["3rd_quartile"])
                    low.append(stat["low"])
                    high.append(stat["high"])

                fig.add_trace(
                    go.Box(
                        y=y,
                        q1=q1,
                        median=median,
                        q3=q3,
                        mean=mean,
                        lowerfence=low,
                        upperfence=high,
                        name=recipe,
                        marker_color=c[test_index],
                        visible=True,
                        boxpoints=False,  # Do not show individual points
                    )
                )
                test_index += 1

            fig.update_traces(
                orientation="h",
                boxmean=True,
                jitter=1,
            )
            fig.update_layout(
                boxmode="group",
                height=height,
                width=900,
                title=dict(
                    text=dataset, x=0.03, y=0.03, xanchor="left", yanchor="bottom"
                ),
                yaxis_title="metric",
                xaxis_title="score",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            write_image(fig, f"./{dataset}_box_plot.png")

    def output_histograms_by_dataset(self, df: pd.DataFrame, metrics: List[str]):
        # Append "latency" to the metrics list
        metrics.append("latency")

        # Get unique datasets
        datasets = df["dataset"].unique()

        for dataset in datasets:
            # Filter DataFrame for the current dataset
            df_filtered = df[df["dataset"] == dataset]

            # Melt the DataFrame to long format
            df_melted = pd.melt(
                df_filtered,
                id_vars=["record_id", "recipe", "dataset"],
                value_vars=metrics,
                var_name="metric",
                value_name="value",
            )

            # Set the theme for the plot
            sns.set_theme(style="darkgrid")

            # Custom function to set bin ranges and filter invalid values
            def custom_hist(data, **kws):
                metric = data["metric"].iloc[0]
                data = data[
                    np.isfinite(data["value"])
                ]  # Remove NaN and infinite values
                data = data[data["value"] >= 0]  # Ensure no negative values
                if metric == "latency":
                    bins = np.concatenate(
                        [
                            np.linspace(
                                0,
                                15,
                            ),
                            [np.inf],
                        ]
                    )  # 46 bins from 0 to 15 seconds, plus one for >15 seconds
                    sns.histplot(data, x="value", bins=bins, stat="percent", **kws)
                else:
                    bin_range = (0, 1)
                    sns.histplot(
                        data,
                        x="value",
                        stat="percent",
                        bins=10,
                        binrange=bin_range,
                        **kws,
                    )

            # Create the FacetGrid
            g = sns.FacetGrid(
                df_melted,
                col="metric",
                row="recipe",
                margin_titles=True,
                height=3.5,
                aspect=1,
                sharex="col",
                legend_out=False,
            )

            g.set_titles(row_template="{row_name}", col_template="{col_name}")

            # Map the custom histogram function to the FacetGrid
            g.map_dataframe(custom_hist)

            for ax, metric in zip(g.axes.flat, g.col_names * len(g.row_names)):
                ax.set_ylim(0, 100)
                # Set custom x-axis label
                if metric == "latency":
                    ax.set_xlabel("Seconds")
                else:
                    ax.set_xlabel("Score")

            g.set_axis_labels(y_var="Percentage")

            # Set the title for the entire figure
            g.figure.suptitle(dataset, fontsize=16)

            # Adjust the layout to make room for the title
            g.figure.subplots_adjust(top=0.9)

            # Save the plot as a PNG file
            g.savefig(f"./{dataset}_histogram_grid.png")

            # Close the plot to avoid displaying it
            plt.close()

    def compare(self, recipes: List[str], output: str):
        df, metrics = self.get_all_data(recipes=recipes)
        if output == "box-plots":
            self.output_box_plots_by_dataset(df=df, metrics=metrics)
        elif output == "histogram-grid":
            self.output_histograms_by_dataset(df=df, metrics=metrics)
        else:
            raise ValueError()

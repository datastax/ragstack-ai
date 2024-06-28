import pandas as pd
import tru_shared

tru = tru_shared.init_tru()

dfAll = pd.DataFrame()

for app in tru.get_apps():
    app_id = app["app_id"]
    print(f"Downloading data for {app_id}...")
    dfRecords, feedbackColumns = tru.get_records_and_feedback([app_id])

    columns_to_keep = feedbackColumns + [
        "record_id",
        "input",
        "output",
        "tags",
        "latency",
        "total_tokens",
        "total_cost",
    ]
    columns_to_drop = [col for col in dfRecords.columns if col not in columns_to_keep]

    dfRecords.drop(columns=columns_to_drop, inplace=True)

    dfRecords["test"] = app_id.split("#")[0]
    dfRecords["test_uuid"] = app_id.split("#")[1]
    dfRecords["dataset"] = app_id.split("#")[2]

    dfAll = pd.concat([dfAll, dfRecords], axis=0, ignore_index=True)

print("Writing results to parquet file.")
dfAll.to_parquet("results.parquet")
print("Done!")

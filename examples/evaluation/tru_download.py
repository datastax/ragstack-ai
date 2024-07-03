import pandas as pd
import tru_shared

tru = tru_shared.init_tru()

df_all = pd.DataFrame()

for app in tru.get_apps():
    app_id = app["app_id"]
    print(f"Downloading data for {app_id}...")
    df_records, feedback_columns = tru.get_records_and_feedback([app_id])

    columns_to_keep = [
        *feedback_columns,
        "record_id",
        "input",
        "output",
        "tags",
        "latency",
        "total_tokens",
        "total_cost",
    ]
    columns_to_drop = [col for col in df_records.columns if col not in columns_to_keep]

    df_records = df_records.drop(columns=columns_to_drop)

    df_records["test"] = app_id.split("#")[0]
    df_records["test_uuid"] = app_id.split("#")[1]
    df_records["dataset"] = app_id.split("#")[2]

    df_all = pd.concat([df_all, df_records], axis=0, ignore_index=True)

print("Writing results to parquet file.")
df_all.to_parquet("results.parquet")
print("Done!")

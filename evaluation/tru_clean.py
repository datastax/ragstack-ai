import pandas as pd

dataset_sizes = {
    "blockchain_solana": 58,
    "braintrust_coda_help_desk": 100,
    "covid_qa": 316,
    "evaluating_llm_survey_paper": 276,
    "history_of_alexnet": 160,
    "llama_2_paper": 100,
    "mini_squad_v2": 195,
    "origin_of_covid_19": 24,
    "patronus_ai_financebench": 98,
    "paul_grahman_essay": 44,
    "uber_10k": 822,
}

df = pd.read_parquet("results.parquet")

datasets = df["dataset"].unique()
test_uuids = df["test_uuid"].unique()
tests = df["test"].unique()

for dataset in datasets:
    for test_uuid in test_uuids:
        for test in tests:
            existing_size = len(df[(df["dataset"] == dataset) & (df["test"] == test) & (df["test_uuid"] == test_uuid)])
            expected_size = dataset_sizes[dataset]
            if existing_size > 0 and existing_size < (expected_size * 0.99):
                print(f"Dropping {existing_size} rows for {test}#{test_uuid}#{dataset} expected {expected_size}")
                df = df[~((df["dataset"] == dataset) & (df["test"] == test) & (df["test_uuid"] == test_uuid))]

df = df.reset_index(drop=True)

df.to_parquet("cleaned_results.parquet")

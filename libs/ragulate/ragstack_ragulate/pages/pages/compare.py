import asyncio
import pandas as pd
from typing import Dict, Iterable, Tuple, List, Optional

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from streamlit_pills import pills

from streamlit_extras.switch_page_button import switch_page

from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)

def extract_ground_truth(answer_correctness_calls: List[str]) -> Optional[str]:
    if not answer_correctness_calls:
        return None

    try:
        first_call = answer_correctness_calls[0]
        return first_call.get('meta', {}).get('ground_truth_response')
    except (IndexError, TypeError):
        return None

def extract_contexts(context_relevance_calls: List[str]) -> List[str]:
    contexts: List[str] = []

    for call in context_relevance_calls:
        try:
            contexts.append(call.get('args', {}).get('context'))
        except (IndexError, TypeError):
            pass

    return contexts

# Columns: ['app_id', 'app_json', 'type', 'record_id', 'input', 'output',
    # 'tags', 'record_json', 'cost_json', 'perf_json', 'ts', 'context_relevance',
    # 'answer_relevance', 'answer_correctness', 'groundedness', 'context_relevance_calls',
    # 'answer_relevance_calls', 'answer_correctness_calls', 'groundedness_calls',
    # 'latency', 'total_tokens', 'total_cost']

def combine_and_calculate_diff(df_list: List[pd.DataFrame], titles: List[str]) -> pd.DataFrame:
    # Ensure the lengths of df_list and titles match
    assert len(df_list) == len(titles), "Number of dataframes and titles must match."

    columns_to_drop = ['app_id', 'app_json', 'type', 'record_id', 'latency',
                       'tags', 'record_json', 'cost_json', 'perf_json', 'ts',
                       'context_relevance_calls', 'answer_relevance_calls',
                       'answer_correctness_calls', 'groundedness_calls', 'total_cost']

    # for call in df_list[0].loc[0,'context_relevance_calls']:
    #     st.write(call)

    # Process the first dataframe to extract ground truth
    df_list[0]['ground_truth'] = df_list[0]['answer_correctness_calls'].apply(extract_ground_truth)
    df_list[0]['contexts'] = df_list[0]['context_relevance_calls'].apply(extract_contexts)
    df_list[0].drop(columns=columns_to_drop, inplace=True)
    df_list[0].columns = [f'{col}_{titles[0]}' if col != 'input' and col != 'ground_truth' else col for col in df_list[0].columns]

    # Process the remaining dataframes
    for df, title in zip(df_list[1:], titles[1:]):
        df['contexts'] = df['context_relevance_calls'].apply(extract_contexts)
        df.drop(columns=columns_to_drop, inplace=True)
        df.columns = [f'{col}_{title}' if col != 'input' else 'input' for col in df.columns]

    # Combine dataframes on 'input' column
    combined_df = df_list[0]
    for df in df_list[1:]:
        combined_df = combined_df.merge(df, on='input', how='outer')

    # If there are exactly two dataframes, calculate the differences
    if len(df_list) == 2:
        for col in ['context_relevance', 'answer_relevance', 'answer_correctness', 'groundedness', 'total_tokens']:
            combined_df[f'{col}__diff'] = combined_df[f'{col}_{titles[0]}'] - combined_df[f'{col}_{titles[1]}']

    # Reorder the columns
    output_columns = [f'output_{title}' for title in titles]
    remaining_columns = sorted([col for col in combined_df.columns if col not in ['input', 'ground_truth'] + output_columns])

    combined_df = combined_df[['input', 'ground_truth'] + output_columns + remaining_columns]

    return combined_df

#@st.cache_data
def get_data(recipes: List[str], dataset: str, timestamp: int) -> Dict[str, List[str]]:
    df_list: List[pd.DataFrame] = []
    for recipe in recipes:
        tru = get_tru(recipe_name=recipe)
        df, feedbacks = tru.get_records_and_feedback(app_ids=[dataset])
        df_list.append(df)

        tru.delete_singleton()

    return combine_and_calculate_diff(df_list=df_list, titles=recipes)


st.title("Compare")
selected_recipes = st.session_state.selected_recipes
dataset = "vcg"
st.write("Selected recipes: " + ", ".join(selected_recipes))
st.write(f"Dataset: {dataset}")
st.dataframe(get_data(recipes=selected_recipes, dataset=dataset, timestamp=0))


if st.button("home"):
    switch_page("home")

import asyncio
import pandas as pd
import json
from typing import Dict, Iterable, Tuple, List, Optional, Any

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

PAGINATION_SIZE = 10

st.set_page_config(page_title="Ragulate - Compare",
                   layout="wide",
                   initial_sidebar_state="collapsed")


def get_tru(recipe_name: str) -> Tru:
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)

def split_into_dict(text:str, keys: List[str]) -> Dict[str,str]:
    # Create a dictionary to hold the results
    result_dict = {}

    # Start with the full text
    remaining_text = text

    # Iterate over the keys
    for i, key in enumerate(keys):
        # Find the start position of the current key
        key_position = remaining_text.find(key + ":")
        if key_position == -1:
            continue

        # Find the end position of the current value
        if i < len(keys) - 1:
            next_key_position = remaining_text.find(keys[i + 1] + ":")
        else:
            next_key_position = len(remaining_text)

        # Extract the value for the current key
        value = remaining_text[key_position + len(key) + 1:next_key_position].strip()

        # Add the key-value pair to the dictionary
        result_dict[key] = value

        # Update the remaining text
        remaining_text = remaining_text[next_key_position:]

    return result_dict

def extract_ground_truth(answer_correctness_calls: List[str]) -> Optional[str]:
    if not answer_correctness_calls:
        return None

    try:
        first_call = answer_correctness_calls[0]
        return first_call.get('meta', {}).get('ground_truth_response')
    except (IndexError, TypeError):
        return None

def extract_contexts(record_json: List[str]) -> List[Any]:
    record = json.loads(record_json)
    calls = record.get("calls", [])
    for call in calls:
        returns = call.get("rets", {})
        if isinstance(returns, dict) and "context" in returns:
            return returns["context"]
    return []

def extract_answer_relevance_reason(answer_relevance_calls: List[str]) -> Optional[Dict[str,str]]:
    if not answer_relevance_calls:
        return None

    try:
        first_call = answer_relevance_calls[0]
        reason = first_call.get('meta', {}).get('reason')
        return split_into_dict(reason, ["Criteria", "Supporting Evidence"])
    except (IndexError, TypeError):
        return None

def extract_context_relevance_reasons(context_relevance_calls: List[str]) -> Optional[List[Dict["str", Any]]]:
    reasons = []
    if isinstance(context_relevance_calls, list):
        for call in context_relevance_calls:
            reason = call.get('meta', {}).get('reason')
            reasons.append({
                "context": call.get('args', {}).get('context'),
                "score": call.get('ret'),
                "reason": split_into_dict(reason, ["Criteria", "Supporting Evidence"])
            })
    return reasons if len(reasons) > 0 else None


def extract_groundedness_reasons(groundedness_calls: List[str]) -> Optional[Dict["str", Any]]:
    if not groundedness_calls:
        return None

    try:
        first_call = groundedness_calls[0]

        return {
            "contexts": first_call.get('args', {}).get('source', []), # list of contexts
            # string with format: `STATEMENT {n}:\nCriteria: {reason}\nSupporting Evidence: {evidence}\nScore: {score}`
            # where n doesn't seem to match with the number of contexts well.
            "reasons": first_call.get('meta', {}).get('reasons')
        }
    except (IndexError, TypeError):
        return None


numericColumnType = [
  "numericColumn",
  "numberColumnFilter"
]

class Column:
    field: str
    children: Dict[str, "Column"]
    type: List[str]
    hide: bool
    width: int
    style: Dict[str, str]

    def __init__(self, field: Optional[str] = None, children: Optional[Dict[str, "Column"]] = None, type: Optional[List[str]] = None, hide: Optional[bool] = False, width: Optional[int] = 0, style: Optional[Dict[str, str]] = None):
        self.field = field
        self.children = children if children is not None else {}
        self.type = type if type is not None else []
        self.hide = hide
        self.width = width if width is not None else 0
        self.style = style

    def get_props(self, headerName: str) -> Dict[str, Any]:
        props:Dict[str, Any] = {
            "headerName": headerName,
            "field": self.field,
            "type": self.type,
        }
        if self.hide:
            props["hide"] = True
        if self.width > 0:
            props["width"] = self.width
        if self.style is not None:
            props["cellStyle"] = {k:v for k,v in self.style.items()}
        return props


def get_column_defs(columns: Dict[str, Column]) -> List[Dict[str, Any]]:
    columnDefs: List[Dict[str, Any]] = []

    for headerName, column in columns.items():
        if len(column.children) == 0:
            columnDefs.append(column.get_props(headerName=headerName))
        else:
            columnDefs.append({
                "headerName": headerName,
                "children": get_column_defs(columns=column.children)
            })
    return columnDefs



def find_common_strings(list_of_lists: List[List[str]]) -> List[str]:
    # Convert each list to a set
    sets = [set(lst) for lst in list_of_lists]

    # Find the intersection of all sets
    common_strings = set.intersection(*sets)

    # Convert the set back to a list (if needed)
    return list(common_strings)

def find_full_set_of_strings(list_of_lists: List[List[str]]) -> List[str]:
    # Convert each list to a set
    sets = [set(lst) for lst in list_of_lists]

    # Find the union of all sets
    full_set_of_strings = set.union(*sets)

    # Convert the set back to a list (if needed)
    return list(full_set_of_strings)

# Columns: ['app_id', 'app_json', 'type', 'record_id', 'input', 'output',
    # 'tags', 'record_json', 'cost_json', 'perf_json', 'ts', 'context_relevance',
    # 'answer_relevance', 'answer_correctness', 'groundedness', 'context_relevance_calls',
    # 'answer_relevance_calls', 'answer_correctness_calls', 'groundedness_calls',
    # 'latency', 'total_tokens', 'total_cost']

def combine_and_calculate_diff(df_list: List[pd.DataFrame], feedbacks_list: List[List[str]], recipes: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure the lengths of df_list and recipes match
    assert len(df_list) == len(recipes), "Number of dataframes and recipes must match."

    feedbacks = find_common_strings(feedbacks_list)

    columns_to_drop = [ 'app_id', 'app_json', 'type', 'record_id', 'latency',
                        'tags', 'record_json', 'cost_json', 'perf_json', 'ts','total_cost']

    for feedback in find_full_set_of_strings(feedbacks_list):
        columns_to_drop.append(f"{feedback}_calls")

    columns_to_diff = feedbacks + ["total_tokens"]


    # st.json(df_list[0].loc[0,'groundedness_calls'])

    for i, (df, recipe) in enumerate(zip(df_list, recipes)):
        if i == 0:
            df['ground_truth'] = df['answer_correctness_calls'].apply(extract_ground_truth)
        df['answer_relevance_reason'] = df["answer_relevance_calls"].apply(extract_answer_relevance_reason)
        df['context_relevance_reasons'] = df["context_relevance_calls"].apply(extract_context_relevance_reasons)
        df['groundedness_reasons'] = df["groundedness_calls"].apply(extract_groundedness_reasons)
        df['contexts'] = df['record_json'].apply(extract_contexts)
        df.drop(columns=columns_to_drop, inplace=True)
        df.columns = [f'{col}_{recipe}' if col not in ['input', 'ground_truth'] else col for col in df.columns]

    combined_df = df_list[0]
    for df in df_list[1:]:
        combined_df = combined_df.merge(df, on='input', how='outer')

    # If there are exactly two dataframes, calculate the differences
    if len(df_list) == 2:
        for col in columns_to_diff:
            combined_df[f'{col}__diff'] = combined_df[f'{col}_{recipes[0]}'] - combined_df[f'{col}_{recipes[1]}']

    # Reorder the columns
    output_columns = [f'output_{recipe}' for recipe in recipes]
    remaining_columns = sorted([col for col in combined_df.columns if col not in ['input', 'ground_truth'] + output_columns])

    combined_df = combined_df[['input', 'ground_truth'] + output_columns + remaining_columns]

    return (combined_df, columns_to_diff)

#@st.cache_data
def get_data(recipes: List[str], dataset: str, timestamp: int) -> Tuple[pd.DataFrame, List[str]]:
    df_list: List[pd.DataFrame] = []
    feedbacks_list: List[List[str]] = []
    for recipe in recipes:
        tru = get_tru(recipe_name=recipe)
        df, feedbacks = tru.get_records_and_feedback(app_ids=[dataset])
        df_list.append(df)
        feedbacks_list.append(feedbacks)
        tru.delete_singleton()

    return combine_and_calculate_diff(df_list=df_list, feedbacks_list=feedbacks_list, recipes=recipes)

if st.button("home"):
    switch_page("home")

recipes = st.session_state.selected_recipes
dataset = "vcg"
compare_df, data_cols = get_data(recipes=recipes, dataset=dataset, timestamp=0)

# st.write(compare_df.columns.tolist())

columns: Dict[str, Column] = {}

columns["Query"] = Column(field="input", style={"word-break": "break-word"})
columns["Answer"] = Column()

for recipe in recipes:
    columns["Answer"].children[recipe] = Column(field=f"output_{recipe}", width=400, style={"word-break": "break-word"})
    columns[f"contexts_{recipe}"] = Column(field=f"contexts_{recipe}", hide=True)
    columns[f"answer_relevance_reason_{recipe}"] = Column(field=f"answer_relevance_reason_{recipe}", hide=True)
    columns[f"context_relevance_reasons_{recipe}"] = Column(field=f"context_relevance_reasons_{recipe}", hide=True)
    columns[f"groundedness_reasons_{recipe}"] = Column(field=f"groundedness_reasons_{recipe}", hide=True)

columns["Answer"].children["Ground Truth"] = Column(field="ground_truth", width=400, style={"word-break": "break-word"})

for data_col in data_cols:
    columns[data_col] = Column()
    for recipe in recipes:
        columns[data_col].children[recipe] = Column(field=f"{data_col}_{recipe}", type=numericColumnType, width=(len(recipe)*7)+ 50)
    if len(recipes) == 2:
        columns[data_col].children["Diff"] = Column(field=f"{data_col}__diff", type=numericColumnType, width=(len("Diff")*7)+ 50)


gb = GridOptionsBuilder.from_dataframe(compare_df)

gb.configure_default_column(autoHeight=True, wrapText=True)
gb.configure_pagination(paginationPageSize=PAGINATION_SIZE, paginationAutoPageSize=False)
gb.configure_side_bar()
gb.configure_selection(selection_mode="single", use_checkbox=False)

gridOptions = gb.build()
gridOptions["columnDefs"] = get_column_defs(columns=columns)
data = AgGrid(
    compare_df,
    gridOptions=gridOptions,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
)

selected_rows = data.selected_rows
selected_rows = pd.DataFrame(selected_rows)

if len(selected_rows) == 0:
    st.write("Hint: select a row to display details of a record")

else:
    # Start the record specific section
    st.divider()

    st.subheader(f"Query")
    st.caption(selected_rows['input'][0])
    st.subheader(f"Ground Truth")
    st.caption(selected_rows['ground_truth'][0])

    table = {}
    for recipe in recipes:
        column_data = [selected_rows[f"output_{recipe}"][0]]
        for data_col in data_cols:
            column_data.append(selected_rows[f"{data_col}_{recipe}"][0])
        table[recipe] = column_data

    context_indexes: Dict[str, Dict[str, int]] = {}

    df = pd.DataFrame(table)
    df.index = ["Answer"] + data_cols
    st.subheader(f"Results")
    st.table(df)

    st.subheader(f"Contexts")
    context_cols = st.columns(len(recipes))
    for i, recipe in enumerate(recipes):
        context_indexes[recipe] = {}
        for j, context in enumerate(selected_rows[f"contexts_{recipe}"][0]):
            context_cols[i].caption(f"Chunk: {j + 1}")
            context_indexes[recipe][context["page_content"]] = j
            with context_cols[i].popover(f"{json.dumps(context['page_content'][0:200])}..."):
                st.caption("Metadata")
                st.json(context["metadata"], expanded=False)
                st.caption("Content")
                st.write(context["page_content"])

    st.subheader(f"Reasons")

    with st.expander(f"Answer Relevance"):
        reason_cols = st.columns(len(recipes))
        for i, recipe in enumerate(recipes):
            reason_cols[i].caption(f"")
            reason_cols[i].json(selected_rows[f"answer_relevance_reason_{recipe}"][0])


    with st.expander(f"Context Relevance"):
        reason_cols = st.columns(len(recipes))
        for i, recipe in enumerate(recipes):
            context_reasons: Dict[int, Dict[str, Any]] = {}
            for context_reason in selected_rows[f"context_relevance_reasons_{recipe}"][0]:
                context_index = context_indexes[recipe][context_reason["context"]]
                context_reasons[context_index] = {
                    "score": context_reason["score"],
                    "reason": context_reason["reason"],
                }
            reason_cols[i].json(context_reasons)

    with st.expander(f"Groundedness"):
        reason_cols = st.columns(len(recipes))
        for i, recipe in enumerate(recipes):
            groundedness_reasons: Dict[str, Any] = {"contexts": [], "reasons": []}
            for context in selected_rows[f"groundedness_reasons_{recipe}"][0]["contexts"]:
                groundedness_reasons["contexts"].append(context_indexes[recipe][context])
            groundedness_reasons["reasons"] = selected_rows[f"groundedness_reasons_{recipe}"][0]["reasons"].split("\n\n")
            reason_cols[i].json(groundedness_reasons)

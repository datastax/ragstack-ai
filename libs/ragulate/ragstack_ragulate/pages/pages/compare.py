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

    columns_to_drop = ['app_id', 'app_json', 'type', 'record_id', 'latency', 'tags',
                       'cost_json', 'perf_json', 'ts','total_cost']

    for feedback in find_full_set_of_strings(feedbacks_list):
        columns_to_drop.append(f"{feedback}_calls")

    columns_to_diff = feedbacks + ["total_tokens"]

    # for call in df_list[0].loc[0,'context_relevance_calls']:
    #     st.write(call)

    # Process the first dataframe to extract ground truth
    df_list[0]['ground_truth'] = df_list[0]['answer_correctness_calls'].apply(extract_ground_truth)
    df_list[0]['contexts'] = df_list[0]['context_relevance_calls'].apply(extract_contexts)
    df_list[0].drop(columns=columns_to_drop, inplace=True)
    df_list[0].columns = [f'{col}_{recipes[0]}' if col != 'input' and col != 'ground_truth' else col for col in df_list[0].columns]

    # Process the remaining dataframes
    for df, recipe in zip(df_list[1:], recipes[1:]):
        df['contexts'] = df['context_relevance_calls'].apply(extract_contexts)
        df.drop(columns=columns_to_drop, inplace=True)
        df.columns = [f'{col}_{recipe}' if col != 'input' else 'input' for col in df.columns]

    # Combine dataframes on 'input' col
    # umn
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

@st.cache_data
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

columns: Dict[str, Column] = {}

columns["Query"] = Column(field="input", style={"word-break": "break-word"})
columns["Answer"] = Column()

for recipe in recipes:
    columns["Answer"].children[recipe] = Column(field=f"output_{recipe}", width=400, style={"word-break": "break-word"})
    columns[f"contexts_{recipe}"] = Column(field=f"contexts_{recipe}", hide=True)

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

    df = pd.DataFrame(table)
    df.index = ["Answer"] + data_cols
    st.subheader(f"Results")
    st.table(df)

    st.subheader(f"Contexts")
    context_cols = st.columns(len(recipes))
    for i, recipe in enumerate(recipes):
        for j, context in enumerate(selected_rows[f"contexts_{recipe}"][0]):
            context_cols[i].caption(f"Chunk: {j + 1}")
            with context_cols[i].popover(f"{json.dumps(context[0:200])}..."):
                st.write(context)







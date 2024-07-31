
import asyncio
import os
import glob

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

from millify import millify
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from typing import Dict, List

from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)

# Define CSS for the card layout
card_css = """
<style>
.card-container {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.card-title {
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
"""

# Apply the CSS to the Streamlit app
st.markdown(card_css, unsafe_allow_html=True)

def start_card(title: str):
    st.markdown(f'<div class="card-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)

def end_card():
    st.markdown('</div>', unsafe_allow_html=True)


# Create a function to display a card-like container with Streamlit widgets
def display_card(title, widget_func):
    with st.container():

        widget_func()


@st.cache_data
def get_recipes(timestamp: int) -> Dict[str, List[str]]:
    recipes = {}

    for file in glob.glob(os.path.join("*.sqlite")):
        recipe = file.removesuffix(".sqlite")
        datasets = []

        tru = get_tru(recipe_name=recipe)

        for app in tru.get_apps():
            datasets.append(app["app_id"])

        recipes[recipe] = datasets

        tru.delete_singleton()

    return recipes


if __name__ == "__main__":
    if 'recipe_cache_time' not in st.session_state:
        st.session_state.recipe_cache_time = 0

    if 'selected_recipes' in st.session_state:
        for selected_recipe in st.session_state.selected_recipes:
            recipe_key = f"toggle_{selected_recipe}"
            if recipe_key not in st.session_state:
                st.session_state[recipe_key] = True

def get_recipe_state(recipe: str) -> bool:
    if 'selected_recipes' in st.session_state:
        return recipe in st.session_state.selected_recipes
    return False


def home():
    """Render the home page."""
    # compare = st.Page("compare.py", title="Compare")

    # st.navigation(pages=[compare], position="hidden")

    st.title("Recipe List")
    st.write(
        "Select Recipes and Datasets to Compare..."
    )

    selected_recipes = []

    for recipe, datasets in get_recipes(st.session_state.recipe_cache_time).items():
        recipe_key = f"toggle_{recipe}"
        #start_card(recipe)
        st.toggle(label=recipe, key=recipe_key)
        st.radio(label=recipe, options=datasets, key=recipe, label_visibility="hidden")
        #end_card()

        if st.session_state[recipe_key] == True:
            selected_recipes.append(recipe)

    st.write("Selected recipes: " + ", ".join(selected_recipes))

    if st.button("Compare", key="button_compare", disabled=len(selected_recipes)<2):
        st.session_state.selected_recipes = selected_recipes
        switch_page("compare")


if __name__ == "__main__":
    home()
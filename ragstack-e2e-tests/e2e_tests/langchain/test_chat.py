import logging
import tempfile

import pytest
from e2e_tests.conftest import (
    set_current_test_info,
)
from langchain_community.chat_models import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def set_test_info(chat: str):
    set_current_test_info("langchain::chat", chat)


@pytest.fixture
def vertex_gemini():
    return ChatVertexAI(model_name="gemini-pro")


@pytest.mark.parametrize(
    "chat",
    ["vertex_gemini"],
)
def test_chat(chat, request):
    set_test_info(chat)
    chat_model = request.getfixturevalue(chat)
    prompt = ChatPromptTemplate.from_messages([("human", "Hello! Where Archimede was born?")])
    chain = prompt | chat_model
    response = chain.invoke({})
    assert "Syracuse" in response.content

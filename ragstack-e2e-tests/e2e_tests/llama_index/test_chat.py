from e2e_tests.conftest import (
    set_current_test_info,
)

# from llama_index.llms import Gemini


def set_test_info(chat: str):
    set_current_test_info("llama_index::chat", chat)


# @pytest.fixture
# def gemini():
#     return Gemini(model="models/gemini-pro")


# @pytest.mark.parametrize(
#     "chat",
#     [
#         # "gemini",
#     ],
# )
# def test_chat(chat, request):
#     set_test_info(chat)
#     chat_model = request.getfixturevalue(chat)
#     response = chat_model.complete("Hello! Where Archimede was born?")
#     assert "Syracuse" in response.content

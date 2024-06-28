import os

import openai
from flask import Flask
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
@app.route("/{city}")
def hello():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["city"],
        template="Plan one day tour in {city}?",
    )
    question = prompt.format(city="New York")
    result = llm(question)
    response = app.response_class(
        response=result,
        status=200,
        mimetype="text/plain",
    )
    response.mimetype = "text/plain"

    return response


# Execute the application
if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")

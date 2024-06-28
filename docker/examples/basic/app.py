import os

import openai
from flask import Flask
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
@app.route("/<city>")
def hello(place="city"):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=['city'],
        template="Plan one day tour in {city}?",
    )
    question = prompt.format(city=city)
    result = llm(question)
    response = make_response(result.content, requests.codes.ok)
    response.mimetype = "text/plain"

    return response

# Execute the application
if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")

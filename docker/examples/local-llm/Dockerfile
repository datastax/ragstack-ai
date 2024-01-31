# Use an official Python runtime as a base image
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt /app

# Install Python dependencies from the requirements file
RUN pip install -r requirements.txt

# Download and save the model and tokenizer to a directory
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('bert-base-uncased'); tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); model.save_pretrained('./local_model_directory'); tokenizer.save_pretrained('./local_model_directory')"

COPY . .

CMD ["python", "./app.py"]


from langchain.chains import SimpleChain
from langchain.transformers import TransformersWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the local model and tokenizer
model = AutoModelForCausalLM.from_pretrained('./local_model_directory')
tokenizer = AutoTokenizer.from_pretrained('./local_model_directory')

# Wrap the model with LangChain's TransformersWrapper
model_wrapper = TransformersWrapper(model=model, tokenizer=tokenizer)

# Create a LangChain chain with the model
chain = SimpleChain(model_wrapper)

# Use the chain for some task, e.g., generating text
input_text = "Once upon a time"
generated_text = chain.run(input_text)
print(generated_text)


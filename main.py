import os
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.7, "max_length": 200},
    huggingfacehub_api_token=hf_api_token
)

def ask_model(prompt):
    """Send a prompt to the LLM and get a response."""
    response = llm(prompt)
    return response

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    answer = ask_model(prompt)
    print(f"\nLLM Response: {answer}")
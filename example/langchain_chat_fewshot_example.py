import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials

load_dotenv()
api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

print("\n------------- Example (LangChain)-------------\n")

params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

langchain_model = LangChainInterface(model="google/flan-t5-xxl", params=params, credentials=creds)
print(langchain_model("Answer this question: What is life?"))


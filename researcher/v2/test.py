import json
from openai import OpenAI

from v2.pipeline import run_pipeline
from dotenv import load_dotenv
load_dotenv("v2/.env")
client = OpenAI()

def llm_call(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-5.6",
        input=prompt,
    )
    return response.output_text

result = run_pipeline(
    "How does STAIR improve document retrieval?",
    llm_call=llm_call,
)

print(json.dumps(result, indent=2, ensure_ascii=False))
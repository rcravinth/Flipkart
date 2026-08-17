import os
from openai import AzureOpenAI

def summarize_text(text: str) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not all([endpoint, api_key, deployment]):
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT."
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    prompt = f"""
Summarize this customer-support interaction for a supervisor.

Return exactly:
Issue:
Customer sentiment:
Resolution/status:
Recommended action:

Interaction:
{text}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise customer-support QA assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=300
    )
    return response.choices[0].message.content

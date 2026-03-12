from utils.llm_client import get_client, get_deployment_name

client = get_client()

completion = client.chat.completions.create(
    model=get_deployment_name(),
    messages=[
        {"role": "system", "content": "You are a helpful tax assistant."},
        {"role": "user", "content": "What is the current corporate tax rate in the United States?"}
    ],
    # temperature=0.7,
    max_completion_tokens=500
)


print(completion.choices[0].message.content)
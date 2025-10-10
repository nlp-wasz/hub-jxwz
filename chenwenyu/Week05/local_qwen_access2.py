from openai import OpenAI

client=OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"    #Ollama does not need true api key
)

resp=client.chat.completions.create(
    model="qwen3:0.6b",
    messages=[
        {"role": "system", "content": "You are an helpful mathematics assistant. "},
        {"role": "user", "content": "What is Minkovski Inequity?"}
    ],
    temperature=0.1,
    max_tokens=500
)

print(f"Model:{resp.model}") 
print(f"Content:{resp.choices[0].message.content}")

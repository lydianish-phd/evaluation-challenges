from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);

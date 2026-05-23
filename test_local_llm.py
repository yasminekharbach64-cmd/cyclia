import ollama

response = ollama.chat(
    model='mistral',
    messages=[
        {'role': 'user', 'content': 'Dime en una frase qué es la diabetes tipo 2.'}
    ]
)

print(response['message']['content'])

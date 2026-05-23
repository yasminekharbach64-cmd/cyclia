from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke("Dime en una frase qué es la diabetes tipo 2.")
print(response.content)

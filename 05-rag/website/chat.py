from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore



load_dotenv()
client = OpenAI()


# model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


vecor_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="learning_vectors",
            embedding=embedding_model
)


query = input(">")

search_result = vecor_db.similarity_search(query=query)


parts = []
for result in search_result:
    parts.append(
        f"Page Content: {result.page_content}\n"
        f"Page title: {result.metadata.get('title', 'N/A')}\n"
        f"File Location: {result.metadata.get('source', 'N/A')}"
    )
context = "\n\n\n".join(parts)


SYSTEM_PROMPT = f"""
     You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a websitealong with page_contents and page title.

    You should only ans the user based on the following context and navigate the user
    to open the right page title to know more.

    Context: {context}
"""

chat_completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role":"system","content":context},
              
              {"role":"user", "content":query}]
)

print(f" {chat_completion.choices[0].message.content}")
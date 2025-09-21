from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client= OpenAI()

text = "Hello, This is text for vecotor embading"
response = client.embeddings.create(model="text-embedding-3-small", input=text)

print("Response: ", len(response.data[0].embedding))
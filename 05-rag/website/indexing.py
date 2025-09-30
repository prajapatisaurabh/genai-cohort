from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()




#loading pdf file
loader = WebBaseLoader(["https://docs.chaicode.com/youtube/chai-aur-sql/welcome/","https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/"],
                           header_template={"User-Agent": "my-app/1.0"})
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
)


split_docs = text_splitter.split_documents(docs)

# model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


#vecotor embading
vector_store = QdrantVectorStore.from_documents(
     documents=split_docs,
            url="http://localhost:6333",
            collection_name="learning_vectors",
            embedding=embedding_model
)

print("Indexing of document done")
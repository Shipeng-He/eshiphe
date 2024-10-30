import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import *


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredPowerPointLoader("C:\\AI_Hackathan\\Question 5\\Project decision document 1.pptx")
    docs = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts1 = text_splitter.split_documents(docs)
    print(f"created {len(texts1)} chunks")


    loader = UnstructuredExcelLoader("C:\\AI_Hackathan\\Question 5\\Project financial data 1.xlsx")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts2 = text_splitter.split_documents(docs)
 

embeddings = AzureOpenAIEmbeddings(
    model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"]  # Use the correct version for the LLM Model selected
)



print("vector storing...")
db = FAISS.from_documents((texts1+texts2), embeddings)

print(db.index.ntotal)
# Write our index to disk.
db.save_local("faiss_index")
print("Ingestion is finish")

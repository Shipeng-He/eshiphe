import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import *

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"]  # Use the correct version for the LLM Model selected
)

print("Ingesting...")
loader = UnstructuredPowerPointLoader("C:\\AI_Hackathan\\Question 6\\CU XYZ Change Request Process.pptx")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts1 = text_splitter.split_documents(docs)

loader = UnstructuredExcelLoader("C:\\AI_Hackathan\\Question 6\\Project basic inform.xlsx")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts2 = text_splitter.split_documents(docs)

loader = UnstructuredExcelLoader("C:\\AI_Hackathan\\Question 6\\Project Change Request log.xlsx")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts3 = text_splitter.split_documents(docs)


db = FAISS.from_documents((texts3+texts1+texts2), embeddings)


print(db.index.ntotal)
db.save_local("faiss_index")
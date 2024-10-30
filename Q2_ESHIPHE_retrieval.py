import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
if __name__ == "__main__":
    print("Retrieving...")
    # Initialize Azure OpenAI embeddings with custom model name and correct API version
    embeddings = AzureOpenAIEmbeddings(
            model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],  # Use custom embedding model name from environment variable
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"],  # Correct API version for embedding
            openai_api_type="azure",  # Specify the API type
        )
    # Load the saved FAISS store from the disk.
    db = FAISS.load_local("faiss_index",  embeddings, allow_dangerous_deserialization=True)

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],  # Correct deployment model
        api_version=os.environ["OPENAI_API_VERSION_LLM"],  # Correct API version for deployment model
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),return_source_documents = True
    )

    # Define the query
    query = "Why delay critical path will result into whole project delay?"
    result = qa.invoke({"query":query})
    # Print the result
    print(result)
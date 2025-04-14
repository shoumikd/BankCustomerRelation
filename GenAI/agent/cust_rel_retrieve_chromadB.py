from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

from constants import path_constants



persist_directory = path_constants.PERSIST_DIRECTORY

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding_model)


retriever = vectordb.as_retriever(search_kwargs={"k": 2})


llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=llm_model_name,
    max_length=10,  
    temperature=0.5,  
    max_new_tokens=500
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)


while True:
    user_input = input("\nUser Input: ")
    if (user_input.lower() == 'quit'):
        print("Okay bye!")
        break
    else:
        llm_response = qa_chain(user_input)
        print(llm_response)
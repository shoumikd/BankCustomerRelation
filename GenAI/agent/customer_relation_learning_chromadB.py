import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import 
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings,SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint



file_name = "..\\Bank Customer Relation\\GenAI\\data\\trainingData\\CustomerRelationShipManagementTutorial.pdf"

if not os.path.isfile(os.path.abspath(file_name)):
    print(f"The file {file_name} does not exist.")

   
loader = PyPDFLoader(file_path=os.path.abspath(file_name))
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(len(texts))


embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
instructor_embeddings1 = embedding_model.embed_documents([chunk.page_content for chunk in texts])

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## Here is the mew embeddings being used
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding_model,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

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
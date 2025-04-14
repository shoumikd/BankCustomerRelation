import os
import chromadb
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from chromadb.utils import embedding_functions
from langchain.vectorstores import FAISS
#from PyPDF2 import PdfReader


__collection_name__='customer_relations'
class CustomerRelationLearning:
    def __init__(self):
        self.db = chromadb.Client()

    def get_customer_relation(self, collection_name):
        return self.db.get_or_create_collection(name=collection_name)

    def close(self):
        self.db.close()


#--------------------------------
#Main Program Starts here
#--------------------------------

customerRelationLearning = CustomerRelationLearning()

#Read the document
file_name = "..\\Bank Customer Relation\\GenAI\\data\\trainingData\\CustomerRelationShipManagementTutorial.pdf"

if not os.path.isfile(os.path.abspath(file_name)):
    print(f"The file {file_name} does not exist.")
#else:
#    inputpdf = PdfReader(open(file_name, "rb"))


# Load documents
loader = PyPDFLoader(file_path=os.path.abspath(file_name))
documents = loader.load()

# page_content_text_list=[]
# for page_num in range(4, len(inputpdf.pages)):
#     page = inputpdf.pages[page_num]
#     page_content_text = page.extract_text()
#     page_content_text_list.append(page_content_text)


# Instantiate the text splitter
# text_splited_list=[]
# for page_content_text in page_content_text_list:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False)
#     #Text Splitter
#     # text_splited = text_splitter.split_text(page_content_text)
#     #Document Creator
#     text_splited = text_splitter.create_documents(page_content_text)
#     text_splited_list.extend(text_splited)
# print(f"Number of text chunks generated :{len(text_splited_list)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

#Document Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# doc_splits = text_splitter.split_documents(os.path.abspath(file_name))
#print(f"Number of document chunks generated :{len(doc_splits)}")


# #Instatiate embedding model
# embed_model_name = "all-MiniLM-L6-v2"
# embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])



# Create a FAISS vector store from document embeddings
vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)


#collection = customerRelationLearning.get_customer_relation(__collection_name__)

# collection.add(
#     documents=["This is a document about pineapple","This is a document about oranges"],
#     ids=["id1", "id2"])
# print("Added documents to collection")

# vectorstore = Chroma.from_documents(documents=text_splited,
#                                     embedding=embeddings,
#                                     collection_name="local_rag")


#Querying the collection
# print ("Test collection")
# results = customerRelationLearning.get_customer_relation(__collection_name__).query(
#     query_texts=["This is a query document about india"], # Chroma will embed this for you
#     n_results=2, # how many results to return
#     include=["documents"] # what fields to include in the results
# )
# print(results)



# Instantiate the llm model
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=llm_model_name,
    max_length=10,  
    temperature=0.5,  
    max_new_tokens=500
)

# Example usage (replace with your actual prompt)
# prompt = "What is the capital of France?"
# response = llm.invoke(prompt)
# print("Test Response:",response)

# retriever = vectorstore.as_retriever(search_kwargs={"k":2})
retriever = vectorstore.as_retriever()

prompt = PromptTemplate(
    template="""<|begin_of_text|>
                    <|start_header_id|>system\nYou are an Customer Relation Manager Assistant named Sushi trained on custom Document. 
                    Please use the same training for answering the Queries from user. Make sure to not answer anything not related to the role defined. 
                    You may use your existing knowledge only to a minimal level.
                    <|end_header_id|>
                    <|start_header_id|>user\nWhat is your role
                    <|end_header_id|> 
                    <|start_header_id|>assistant\nI am a helpful assistant and try to give you the best reponse with the related knowledge 
                    I have acquired
                    <|end_header_id|>
                   <|end_of_text|>   
    Question to route: {question} <|start_eot_id|><|start_header_id|>assistant<|end_header_id|><|end_eot_id|>""",
    input_variables=["question"],
)
# start = time.time()
# question_router = prompt | llm | JsonOutputParser()
# #
# question = "What is Customer Relstion manager?"
# print(question_router.invoke({"question": question}))
# end = time.time()
# print(f"The time required to generate response by Router Chain in seconds:{end - start}")


while True:
    user_input = input("\nUser Input: ")
    if (user_input.lower() == 'quit'):
        print("Okay bye!")
        break
    else:
        start = time.time()
        retrieval_grader = prompt | llm #| JsonOutputParser()
        question = user_input
        docs = retriever.invoke(question)
        #doc_txt = docs[1].page_content
        #print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
        print(retrieval_grader.invoke({"question": question}))
        end = time.time()
        print(f"The time required to generate response by the retrieval grader in seconds:{end - start}")


#############################RESPONSE ###############################
#{'datasource': 'vectorstore'}
#The time required to generate response by Router Chain in seconds:0.34175705909729004
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import 
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings,SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from constants import path_constants



file_name = path_constants.TRAINING_DATA_PATH+"\\CustomerRelationShipManagementTutorial.pdf"


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
persist_directory = path_constants.PERSIST_DIRECTORY

## Here is the mew embeddings being used
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding_model,
                                 persist_directory=persist_directory)

# persist the db to disk
vectordb.persist()
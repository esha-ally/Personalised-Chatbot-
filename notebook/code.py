from langchain.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import CTransformers, Ollama
from sentence_transformers import SentenceTransformer
from torch._C import *

##from dotenv import load_dotenv
import os
##load_dotenv()
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
#DATA INGESTION
def load_pdf(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs
#pip install pypdf
extracted_data = load_pdf(data=r"C:\Users\rahul\Desktop\CV\7th sem")
len(extracted_data)
#DATA TRANSFORMATION
#CREATING TEXT CHUNKS
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
docs = text_split(extracted_data=extracted_data)
len(docs)
#Use embedding model
def hugging_face_embed_docs():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def ollama_embed_docs():
    embeddings = OllamaEmbeddings(model="llama2")
    return embeddings
#pip install sentence-transformers
embeddings = ollama_embed_docs()
hugging_embeddings = hugging_face_embed_docs()
embeddings
hugging_embeddings
#pip install faiss-cpu


from langchain.vectorstores import FAISS
vector_store = FAISS.from_documents(docs, hugging_embeddings)

vector_store
sample_ans = vector_store.similarity_search("What is an allergy", k=3)
sample_ans
prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If you know the answer, answer it in a crisp and descriptive way. Use around 5 to 6 sentences to explain.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    At the end of each answer which you know, don't end it with any statement thanking the user for asking you the question. Just end your answer normally without thanking the user. Just anser what is necessary. No thank you to the user is needed
    If the user tells "Thank you", or anything related to it, then reply in short, thanking the user to ask questions to you and telling him to consult you again in the future in case of any help.
    Assistant : 
    """
PROMPT = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt" : PROMPT}
# llm = CTransformers(model = r"C:\Users\Arjo\.ollama\models\blobs\sha256-8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246", model_type='llama2', config={'max_new_tokens' : 512, 'temperature' : 0.8})
llm = Ollama(model="phi3:mini", temperature=0.8, num_predict=1024)
qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    chain_type_kwargs = chain_type_kwargs,
    return_source_documents = True,
    retriever = vector_store.as_retriever(search_kwargs = {'k' : 2})
)

while True:
    user_input = input(f"Input Prompt : ")
    result = qa({'query' : user_input})
    print("Response : ", result['result'])

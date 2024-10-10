import os
import PyPDF2
from langchain.schema import Document
# import nltk

# nltk.download("punkt_tab")

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

from langchain_community.embeddings import (
    OllamaEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.vectorstores import FAISS  # Import FAISS
import numpy as np
import faiss
from langchain.docstore import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


def load_pdfs_with_pypdf2(folder_path):
    data = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            pdf_path = os.path.join(root, filename)
            try:
                with open(pdf_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in range(len(reader.pages)):
                        text += reader.pages[page].extract_text()
                    data.append(text)
            except Exception as err:
                print(f"Could not read this file: {pdf_path} \n {str(err)}")
    return data


def load_files(folder_path):
    data = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            try:
                pdf_path = os.path.join(root, filename)
                loader = UnstructuredPDFLoader(file_path=pdf_path)
                data += loader.load()
                print(data)
            except Exception as err:
                print(f"Could not read this file: {pdf_path} \n {str(err)}")
    return data


def split_text_into_sentences(text):
    tokens = nltk.word_tokenize(text)
    chunks = nltk.sent_tokenize(text, tokens)
    return chunks


def split_text_with_overlap(text):
    # Split and chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    # Ensure text is a single string
    if isinstance(text, list):
        text = ' '.join(text)  # Join list elements into a single string if needed

    # Wrap the text in a Document object
    documents = [Document(page_content=text)]
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_chroma_vector_embeddings(text):
    chunks = split_text_with_overlap(text)
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # Add to vector database
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="local-rag",
    )


def get_vector_embeddings(text):
    # Split the text into chunks
    chunks = split_text_with_overlap(text)

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # Generate embeddings for the chunks
    chunk_embeddings = embeddings.embed_documents([doc.page_content for doc in chunks])  # Extracting page_content

    # Convert chunk_embeddings to a NumPy array for FAISS
    chunk_embeddings = np.array(chunk_embeddings)  # Convert list of lists to NumPy array

    # Create an in-memory document store
    docstore = InMemoryDocstore()  # Initialize the document store

    # Create the FAISS index
    dim = chunk_embeddings.shape[1]  # Dimension of the embedding
    index = faiss.IndexFlatL2(dim)  # Initialize a flat L2 index

    # Initialize the FAISS vector store with the index and document store
    faiss_index = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,  # Pass the initialized index
        docstore=docstore,  # Pass the document store
        index_to_docstore_id={}  # Empty mapping for docstore IDs
    )

    # Prepare the texts and corresponding IDs for adding
    texts = [doc.page_content for doc in chunks]
    ids = [f"doc_{i}" for i in range(len(texts))]  # Generate unique IDs for each document

    # Add documents and their embeddings to the index
    faiss_index.add_texts(texts=texts, embeddings=chunk_embeddings.tolist(), ids=ids)  # Ensure embeddings is in list format

    return faiss_index


def retrieval(vector_db):
    # LLM from Ollama
    local_model = "mistral"
    llm = ChatOllama(model=local_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate a specified
        number of flashcards, prioritizing the most relevant documents retrieved from a vector database.
        Your goal is to create question-and-answer pairs to make the information easier to study.
        Provide these flashcards in the format 'Question:' followed by 'Answer:', with each pair separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke("Give me 10 flashcards")
    return result


if __name__ == "__main__":
    # text = load_files("assets")
    text = load_pdfs_with_pypdf2("assets")
    if len(text) == 0:
        raise ValueError("data is empty")
    vector_db = get_vector_embeddings(text)
    response = retrieval(vector_db)
    print(response)

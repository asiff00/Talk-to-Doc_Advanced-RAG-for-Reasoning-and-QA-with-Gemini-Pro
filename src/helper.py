from pypdf import PdfReader
import chromadb
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

def _read_pdf(filename):
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts

def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1600, chunk_overlap=200
    )
    character_split_texts = character_splitter.split_text("\n\n".join(texts))
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=20, tokens_per_chunk=300
    )
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
    return token_split_texts

def load_chroma(filename, collection_name, embedding_function):
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )
    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)
    return chroma_collection

def word_wrap(string, n_chars=72):
    if len(string) < n_chars:
        return string
    else:
        return (
            string[:n_chars].rsplit(" ", 1)[0]
            + "\n"
            + word_wrap(string[len(string[:n_chars].rsplit(" ", 1)[0]) + 1 :], n_chars)
        )

import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

#EMBEDDING_MODEL = "intfloat/multilingual-e5-large" # 2.5 GB of VRAM
# EMBEDDING_MODEL = "hkunlp/instructor-large"  # 1.5 GB of VRAM 
EMBEDDING_MODEL = "hkunlp/instructor-xl" # 5GB

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

DB_DIRECTORY = f"{ROOT_DIRECTORY}/DB-LEARNED-LOCAL-DOCUMENTS"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHROMA_SETTINGS = Settings(is_persistent=True, anonymized_telemetry=False)

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    from langchain.document_loaders import UnstructuredFileLoader, CSVLoader, UnstructuredMarkdownLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    LOCAL_DOCUMENTS_DIRECTORY = f"{ROOT_DIRECTORY}/LOCAL-DOCUMENTS"

    DOCUMENT_LOADERS = {
        ".pdf": UnstructuredFileLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def load_local_document(path):
        _, extension = os.path.splitext(path)
        loader_type = DOCUMENT_LOADERS.get(extension)
        if loader_type:
            print(f"Loading : {path}...\n")
            # instanciate loader of type UnstructuredBaseLoader
            loader = loader_type(path) 
            return loader.load()[0]
        else:
            print(f"File ignored (no loader for extension '{extension}') : {path}\n")
            return None


    # fill an array with local documents paths
    paths=[]
    for root, _, files in os.walk(LOCAL_DOCUMENTS_DIRECTORY):
        for file_name in files:
            paths.append(os.path.join(root, file_name))

    # load documents in a thread pool
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_local_document, path) for path in paths]
    
    # Wait for thread pool result
    documents = [future.result() for future in futures if future.result() is not None]
    print(f"{len(documents)} file(s) loaded\n")

    # split documents into chunks
    print("Splitting documents...\n")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print("Documents splitted...\n")

    # Create embeddings (download it the first time only)
    print(f"Creating embeddings with {DEVICE}...\n")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
    )
    print("Embeddings created\n")

    # Calculate and save embeddings
    print("Saving the vector store...\n")
    Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_DIRECTORY,
        client_settings= CHROMA_SETTINGS,
    )
    print("Documents successfully learned\n")

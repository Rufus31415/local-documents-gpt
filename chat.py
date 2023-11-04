from learn import *
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-70b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-70b-chat.Q4_K_M.gguf"


CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE

N_GPU_LAYERS = 100  # 83 layers for Llama-2-70B
N_BATCH = 512

LLM_DIRECTORY = "./llm"

SYSTEM_PROMPT = """You will use the provided context to answer user questions.
Read the given context before answering questions. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question without any form of courtesy."""


if __name__ == "__main__":

    # reloading learned documents from vectore store
    print("Loading the vector store...\n") 
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE})
    db = Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()
    print("Vector store loaded\n") 


    # Download LLM if needed
    print("Retrieving LLM...\n")
    if not os.path.exists(LLM_DIRECTORY):
        os.mkdir(LLM_DIRECTORY)

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_BASENAME,
        resume_download=True,
        cache_dir=LLM_DIRECTORY,
    )
    print("LLM retrieved\n")


    # Instanciate Large Language Modele
    print("Loading LLM...\n")
    kwargs = {
        "model_path": model_path,
        "n_ctx": CONTEXT_WINDOW_SIZE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_batch": N_BATCH,  
        "verbose": True
    }
    if DEVICE == "cuda":
        kwargs["n_gpu_layers"] = N_GPU_LAYERS  

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(callback_manager=callback_manager, **kwargs)
    llm.streaming= True
    print("LLM loaded\n")

    # Create the prompt and the pipeline
    print("Creating prompt...\n")
    instruction = """
    Context: {context}
    User: {question}"""

    prompt = PromptTemplate(input_variables=["context", "question"], template="[INST]<<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n" + instruction + "[/INST]")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )
    print("Ready for chatting !\n")

    while True:
        query = input("\n> Query your documents : ")

        print("\n\nEvaluating the prompt...\n\n");

        res = qa(query)

        print("\n\n####### LOCAL DOCUMENTS USED TO ANSWER THIS QUERY #######")
        for document in res["source_documents"]:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("###################################################")





import gradio as gr
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.chains import RetrievalQA

# Function to initialize and run the QA chain
def ask_question(question):
    # Load data from the web
    loader = WebBaseLoader("https://science.nasa.gov/mission/mars-exploration-rovers-spirit-and-opportunity/")
    data = loader.load()

    # Set up the embedding model and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=data, embedding=embedding_model)

    # Initialize the LLM model
    llm = ChatOllama(model="llama3.1")

    # Pull the prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

    # Get the result from the QA chain
    result = qa_chain({"query": question})
    return result

# Create the Gradio interface
iface = gr.Interface(
    fn=ask_question, 
    inputs=gr.Textbox(label="Ask a question"), 
    outputs=gr.Textbox(label="Answer"),
    title="Mars Explorations",
    description="Ask questions about the Mars Explorations from NASA."
)

# Launch the app
iface.launch()

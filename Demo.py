import gradio as gr
import sqlite3
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain import hub
import bs4
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
    
    # Check if the result is empty or does not make sense
    if not result or 'result' not in result or result['result'].strip() == "":
        return "I don't know"
    
    return result['result']

# Define custom CSS
css = """
    .gradio-button {
        background-color: green !important;
        border-color: green !important;
        color: white !important;
    }
    .gradio-description {
        text-align: center !important;
    }
    .gradio-markdown {
        text-align: center !important;
    }
"""

# Create the Gradio interface
iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Mars Explorations",
    description=(
        "<div style='text-align: center;'>"
        "<strong>Powered by Llama 3.1</strong><br><br>"
        "Ask questions about the Mars Explorations from NASA"
        "</div>"
    ),
    examples=[
        ["What is Mars?"],
        ["What are the Mars rovers?"],
        ["When did the Mars exploration missions start?"],
        ["What is the latest Mars rover mission?"],
        ["How does Mars' atmosphere differ from Earth's?"]
    ],
    article=(
        "<div style='text-align: center;'>"
        "<img src='https://science.nasa.gov/wp-content/uploads/2024/03/35474_PIA14832.jpg' "
        "style='width: 1200px; height: 210px;' alt='Mars Image'/>"
        "<p><strong>Unlock your curiosity</strong></p>"
        "</div>"
    ),
    theme="default",
    css=css
)

# Launch the app
iface.launch()

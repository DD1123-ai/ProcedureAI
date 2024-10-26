import os
import docx2txt
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import gradio as gr
from PyPDF2 import PdfReader

# Load environment variables and set up API keys
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

def load_documents_with_metadata(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            with open(filepath, 'rb') as file:
                pdf = PdfReader(file)
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    doc = Document(text=text, metadata={'file_name': filename, 'page_number': page_num})
                    documents.append(doc)
        elif filename.endswith('.txt') or filename.endswith('.docx'):
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:  # .docx file
                text = docx2txt.process(filepath)
            sections = text.split('\n\n')  # Assuming sections are separated by double newlines
            for section_num, section in enumerate(sections, 1):
                doc = Document(text=section, metadata={'file_name': filename, 'section': f"Section {section_num}"})
                documents.append(doc)
    return documents

# Load Documents with metadata
documents = load_documents_with_metadata("./Procedures")

# Define LLM and embedding model
llm = NVIDIA(api_key=api_key, model="meta/llama-3.2-3b-instruct")
embed_model = NVIDIAEmbedding(api_key=api_key, model_name="nvdia/nv-embedqa-e5-v5")
Settings.llm = llm
Settings.embed_model = embed_model

# Split the documents into vectors and store them in vectorstore
node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, embed_model=embed_model)

# Create query engine
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

def list_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(('.pdf', '.txt', '.docx')):
            documents.append(filename)
    return "\n".join(documents)

def answer_question(question):
    full_prompt = (
        "Please answer the following question based only on the information provided in the stored procedures. "
        "Do not add any new information. If the answer is not in the procedures, say so. "
        "Ensure that your answer is complete and does not miss any relevant information from the source. "
        "Begin your response by restating the question. "
        "Question: " + question
    )
    
    response = query_engine.query(full_prompt)
    
    answer_text = str(response)  # Convert response to string
    
    # Check if the answer indicates no information was found
    no_info_phrases = [
        "I don't have information",
        "I don't have any information",
        "I don't have specific information",
        "I don't have enough information",
        "I don't have relevant information",
        "The information is not available",
        "The answer is not in the procedures",
        "I couldn't find any information",
        "No information about this",
        "There is no information"
    ]
    
    if any(phrase.lower() in answer_text.lower() for phrase in no_info_phrases):
        return answer_text  # Return only the answer without sources
    
    source_info = []
    if hasattr(response, 'source_nodes') and response.source_nodes:
        for source_node in response.source_nodes:
            doc_name = source_node.node.metadata.get('file_name', 'Unknown document')
            page_number = source_node.node.metadata.get('page_number', 'N/A')
            section = source_node.node.metadata.get('section', 'N/A')
            text_content = source_node.node.text[:100] + "..."  # First 100 characters of the text
            
            source_info.append(f"Document: {doc_name}\nPage/Section: {page_number if page_number != 'N/A' else section}\nContent: {text_content}\n")
    
    if source_info:
        full_response = f"{answer_text}\n\nSources:\n" + "\n".join(source_info)
    else:
        full_response = answer_text
    
    return full_response

# Create a CSVLogger instance for flagging
flagging_callback = gr.CSVLogger()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# iPro: Your Intelligent Procedure Assistant")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(lines=2, label="Enter your question here")
            submit_button = gr.Button("Submit")
        
        with gr.Column(scale=1):
            document_list = gr.Textbox(value=list_documents("./Procedures"), label="Documents in Repository", interactive=False)
    
    answer_output = gr.Textbox(lines=20, label="Answer and Sources")
    
    # Add example questions
    gr.Examples(
        examples=[
            "What are the steps for Learning Enablement?",
            "What is the procedure for Sale Enablement?",
            "How do I request time off?",
            "How to create PDF file?",
        ],
        inputs=question_input,
    )
    
    submit_button.click(
        fn=answer_question, 
        inputs=question_input, 
        outputs=answer_output
    )
    
    # Setup flagging
    gr.Markdown("### Feedback")
    gr.Markdown("If the answer is incorrect or unhelpful, please use the flag button below.")
    flagging_callback.setup([question_input, answer_output], "flagged_data")
    
    # Add flagging button
    flag_button = gr.Button("Flag")
    flag_button.click(
        lambda *args: flagging_callback.flag(args),
        [question_input, answer_output],
        None,
        preprocess=False
    )
    
    # Enable queuing for the entire app
    demo.queue()

# Launch the Gradio interface
demo.launch()
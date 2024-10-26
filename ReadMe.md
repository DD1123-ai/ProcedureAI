#  iPro:  Your Intelligent Procedure Assistant

Project code for the [NVIDIA and LlamaIndex Developer Contest](https://developer.nvidia.com/llamaindex-developer-contest#section-contest-process).

# Links



# Main Tech
- Nvidia Embedding: nvdia/nv-embedqa-e5-v5
- Nvidia NIM:  meta/llama-3.2-3b-instruct
- llama-Index
- Gradio

# Setup
<details>
<summary>Setup details</summary>

##  Requirements:
-Python 3.11.9

Setup virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install llama-index-nvidia llama-index-embeddings-nvidia

```
llama-index-nvidia, llama-index-embeddings-nvidia installed separately to avoid dependency conflicts.

## Environment Variable
Create `.env` file in your code root directory:
```bash
# NVIDIA API KEY
NVIDIA_API_KEY="..."

```
Create `Procedures` folder in your code root directory, and save any procedural documents in the folder for embedding.   It only support txt, docx, and pdf format.

</details>

# Run iPro with Gradio
Now, run your code. If you've written the Python code in a file named app.py, then you would run python app.py from the terminal.

The demo below will open in a browser on http://localhost:7860 if running from a file. If you are running within a notebook, the demo will appear embedded within the notebook.

# How to use iPro
- Type any questions in the Question input field, to ask any information from the procedural documents
- Click Submit when done
- If you need any example help, click any question example below the response field

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text, uploaded_file=None):
    if uploaded_file:
        # Ensure the "docs" directory exists
        if not os.path.exists("docs"):
            os.makedirs("docs")
        
        # Get the filename of the uploaded file
        filename = os.path.join("docs", uploaded_file.name)

        # Save the uploaded PDF file to the "docs" directory
        with open(filename, "wb") as f:
            f.write(uploaded_file.read())



    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=[
                         gr.components.Textbox(lines=7, label="Enter your text"),
                         gr.inputs.File(label="Upload PDF", type="file")
                     ],
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)
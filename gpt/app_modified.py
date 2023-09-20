from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
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

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


# Define a list of relevant keywords based on the CIRPs
relevant_keywords = ['cybersecurity', 'incident', 'response', 'threat', 'vulnerability', 'attack', 'malware', 'ransomware', 'phishing', 'intrusion', 'firewall', 'encryption', 'ddos', 'ids', 'ips', 'zero-day', 'patch', 'exploit', 'authentication', 'authorization', 'compliance', 'data breach', 'forensics', 'endpoint', 'penetration testing', 'ssl', 'vpn', 'password', 'mfa', 'risk assessment', 'log analysis', 'incident analysis', 'cyber threat intelligence', 'command and control', 'exfiltration', 'reconnaissance', 'soc', 'siem', 'apt', 'secure coding', 'cryptography', 'security policy', 'incident management', 'incident response team', 'threat actor', 'cyber hygiene', 'spear phishing', 'social engineering', 'security awareness']

def chatbot(input_text):
    # Check for relevance
    if not any(keyword.lower() in input_text.lower() for keyword in relevant_keywords):
        return "This question is not relevant to the context"
        
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Cyber Incident Response Bot")

index = construct_index("docs")
iface.launch(share=True)
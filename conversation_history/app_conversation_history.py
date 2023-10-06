# Check for relevance
relevant_keywords = [
    'cyber', 'incident', 'response', 'threat', 'security', 'attack', 'DDOS', 'DOS', 'phishing', 'ransomware', 'malware',
    'breach', 'compromise', 'firewall', 'insider', 'honeypot', 'network', 'monitoring', 'backup', 'recovery', 'encryption',
    'virus', 'worm', 'trojan', 'spyware', 'adware', 'rootkit', 'keylogger', 'antivirus', 'clean', 'sensitive', 'data', 'leak',
    'sanitization', 'privacy', 'loss', 'prevention', 'insurance', 'authentication', 'password', 'hashing', 'two-factor', 'multi-factor',
    'intrusion', 'detection', 'system', 'IDS', 'IPS', 'intrusion', 'prevention', 'VPN', 'virtual', 'private', 'network', 'protocol',
    'TCP', 'IP', 'UDP', 'router', 'switch', 'port', 'scan', 'brute', 'force', 'credential', 'stuffing', 'botnet', 'C&C', 'command',
    'control', 'exploit', 'vulnerability', 'patch', 'update', 'secure', 'shell', 'SSH', 'telnet', 'FTP', 'HTTP', 'HTTPS', 'SSL', 'TLS',
    'certificate', 'public', 'key', 'private', 'digital', 'signature', 'PKI', 'infrastructure', 'zero-day', 'payload', 'session', 'hijacking',
    'spoofing', 'ARP', 'address', 'resolution', 'MITM', 'man-in-the-middle', 'DNS', 'domain', 'name', 'server', 'resolver', 'cache', 'poisoning',
    'drive-by', 'download', 'clickjacking', 'CSRF', 'cross-site', 'request', 'forgery', 'XSS', 'cross-site', 'scripting', 'SQL', 'injection',
    'OSI', 'model', 'layer', 'endpoint', 'protection', 'EPP', 'EDR', 'endpoint', 'detection', 'response', 'SOC', 'operations', 'center', 'SIEM',
    'management', 'information', 'event', 'APT', 'advanced', 'persistent', 'threat', 'IoT', 'internet', 'things', 'cloud', 'SaaS', 'PaaS', 'IaaS',
    'virtualization', 'hypervisor', 'container', 'Docker', 'Kubernetes', 'orchestration', 'API', 'application', 'programming', 'interface', 'SDK',
    'software', 'development', 'kit', 'proxy', 'reverse', 'gateway', 'DMZ', 'demilitarized', 'zone', 'NAT', 'address', 'translation', 'penetration',
    'testing', 'pentest', 'red', 'team', 'blue', 'purple', 'CISO', 'chief', 'information', 'officer', 'CIO', 'IT', 'technology', 'LAN', 'WAN', 'MAN',
    'local', 'area', 'network', 'wide', 'metropolitan', 'MAC', 'media', 'access', 'control', 'VPN', 'tunnel', 'split', 'tunneling', 'full', 'cryptography',
    'algorithm', 'symmetric', 'asymmetric', 'RSA', 'AES', 'DES', '3DES', 'block', 'cipher', 'stream', 'PGP', 'pretty', 'good', 'privacy'
]

from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import os

####################################################################################

# Insert OpenAI API key found on the website
os.environ["OPENAI_API_KEY"] = '' #Insert key

####################################################################################

# Define the paths to different document folders based on the selected word sets / playbook
document_folders = {
    "DOS & DDOS": "dos_ddos_docs",
    "Phishing": "phishing_docs",
    "Ransomware": "ransomware_docs",
    "Malware": "malware_docs",
    "Data breach": "data_breach_docs",
    "ICS compromise": "ics_compromise_docs",
}

####################################################################################

#Creating a function that will load the data / knowledge from the provided CIRPS
def construct_index(selected_word_set):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="davinci", max_tokens=num_outputs))

    folder_path = document_folders.get(selected_word_set, "default_docs")  # Use a default folder if the set is not found
    documents = SimpleDirectoryReader(folder_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

####################################################################################

predetermined_word_sets = {
    "DOS & DDOS": ["review", "security", "analyse", "mitigate", "report"],
    "Phishing": ["email", "identify", "malware", "clickbait", "URL", "SOC"],
    "Ransomware": ["encrypt", "ransom", "data", "threat", "decryption"],
    "Malware": ["virus", "worm", "trojan horse", "spyware", "adware", "rootkit", "keylogger", "antivirus", "clean"],
    "Data breach": ["sensitive", "data", "leak", "sanitization", "privacy", "loss prevention", "insurance", "two-factor authentication",],
    "ICS compromise": ["firewall", "insider", "threat", "management", "honeypot", "physical", "security", "network", "monitoring", "backup", "recovery"]
}

####################################################################################


# Function to modify the chatbot's response to be outputted in dot points
def modified_chatbot_response(chatbot_response):
    # Split the response into sentences
    sentences = chatbot_response.split('.')
    
    # Filter out empty strings and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Prefix each sentence with a dot point
    dot_point_response = "\n".join([f"• {sentence}" for sentence in sentences])
    
    return dot_point_response

# Create a chat history to store the conversation
chat_history = []

# Create a set to store identified predetermined words
identified_words = set()

# Creating the chatbot function

def chatbot(input_text, selected_word_set):
    # Check for relevance
    if not any(keyword.lower() in input_text.lower() for keyword in relevant_keywords):
        return "This question is not relevant to the context", "", 0.0, "This question is not relevant to the context"
        
    # If relevant, continue with the existing mechanism
    predetermined_words = predetermined_word_sets.get(selected_word_set, [])

    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # Insert "User:" prefix to the user's input in the conversation history
    chat_history.append(f"User: {input_text}")

    # Generate a response based on the chat history
    response = index.query("\n".join(chat_history), response_mode="compact")

    # Check if predetermined words are found in the chatbot's response and update identified words
    for word in predetermined_words:
        if word.lower() in response.response.lower():
            identified_words.add(word)

    # Calculate the percentage / score of predetermined words found
    num_predetermined_words = len(predetermined_words)
    num_found_words = len(identified_words)
    percentage_found = (num_found_words / num_predetermined_words) * 100

    # Create the list of predetermined words selected
    word_list = "\n".join([f"{word} {'✅' if word in identified_words else '❌'}" for word in predetermined_words])

    # Combine the chatbot response, the predetermined word list, and the response's score
    chatbot_response = modified_chatbot_response(response.response.strip())  # Prefix with "Chatbot:" and format as dot points
    predetermined_word_status = word_list
    percentage_found_text = f"{percentage_found:.2f}%"

    # Append the chatbot's response to the chat history and remove extra lines
    chat_history.append(chatbot_response)
    chat_history.append("")  # Add an empty line for line break between input and output

    # Create a conversation display with a one-line gap between initial conversation and subsequent conversation
    conversation_display = "\n".join(chat_history[:2]) + "\n\n" + "\n".join(chat_history[2:])

    return chatbot_response, predetermined_word_status, percentage_found_text, conversation_display, percentage_found_text, conversation_display


####################################################################################

# Create a Gradio interface with a Textbox for input and a Radio input for selecting word sets
interface1 = gr.Interface(
    fn=chatbot,
    allow_flagging="never",
    inputs=[
        gr.inputs.Textbox(lines=5, label="Chat with the AI"),
        gr.inputs.Radio(
            label="Select Predetermined Word Set",
            choices=list(document_folders.keys()),  # Use the keys as radio button labels
            type="value",  
            default=list(document_folders.keys())[0],  # Set the default selected value
        ),
    ],
    outputs=[
        gr.outputs.Textbox(label="Chatbot Response"),
        gr.outputs.Textbox(label="Scenario Analysis"),
        gr.outputs.Textbox(label="Score (%)"),
        gr.outputs.Textbox(label="Conversation History")
    ],
    title="Custom-trained AI Chatbot",
)

####################################################################################

selected_word_set = list(document_folders.keys())[0]  # Default to the first set
index = construct_index(selected_word_set)
interface1.launch(share=True)


relevant_keywords = ['cyber', 'incident', 'response', 'threat', 'security', 'attack', 'DDOS', 'DOS', 'phishing', 'ransomware']

def chatbot(input_text):
    # Check for relevance
    if not any(keyword.lower() in input_text.lower() for keyword in relevant_keywords):
        return "This question is not relevant to the context", "", 0.0, "This question is not relevant to the context"
        
    # If relevant, continue with the existing mechanism
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    
    # Generate a response based on the input_text
    response = index.query(input_text, response_mode="compact")
    
    # Return the chatbot's response
    chatbot_response = modified_chatbot_response(response.response.strip())  # Prefix with "Chatbot:" and format as dot points
    
    return chatbot_response, "", 0.0, chatbot_response

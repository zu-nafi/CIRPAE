from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import sys
import os

####################################################################################

#Insert OpenAI API key found on website
os.environ["OPENAI_API_KEY"] = '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'

####################################################################################

#Creating a function that will load the data / knowledge from the provided CIRPS
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="davinci", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

####################################################################################

#Creating a dictionary with predefined categories, these will be used to test the chatbot's output accuracy
predetermined_word_sets = {
    "DOS & DDOS": ["review", "security", "analyse", "mitigate", "report"],
    "Phishing": ["email", "identify", "malware", "clickbait", "URL", "SOC"],
    "Ransonware": ["encrypt", "ransom", "data", "threat", "decryption"],
    "Malware": ["virus", "worm", "trojan horse", "spyware", "adware", "rootkit", "keylogger", "antivirus", "clean"],
    "Data breach": ["sensitive", "data", "leak", "sanitization", "privacy", "loss prevention", "insurance", "two-factor authentication",],
    "ICS compromise": ["firewall", "insider", "threat", "management", "honeypot", "physical", "security", "network", "monitoring", "backup", "recovery"]

}

####################################################################################

#Creating the chatbot function
def chatbot(input_text, selected_word_sets):
    # Convert the selected_word_sets list into a single set of predetermined words
    predetermined_words = []
    for selected_word_set in selected_word_sets:
        predetermined_words.extend(predetermined_word_sets[selected_word_set])

    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    
    # Check if predetermined words are found in the chatbot's response
    found_words = []
    for word in predetermined_words:
        if word.lower() in response.response.lower():
            found_words.append(f"{word} ✅")
        else:
            found_words.append(f"{word} ❌")
    
    # Calculate the percentage / score of predetermined words found
    num_predetermined_words = len(predetermined_words)
    num_found_words = sum(1 for word in predetermined_words if word.lower() in response.response.lower())
    percentage_found = (num_found_words / num_predetermined_words) * 100
    
    # Create the list of predetermined words selected
    word_list = "\n".join(found_words)
    
    # Combine the chatbot response, the predetermined word list, and the responses score
    chatbot_response = response.response
    predetermined_word_status = word_list
    percentage_found_text = f"Percentage Found: {percentage_found:.2f}%"
    
    return chatbot_response, predetermined_word_status, percentage_found_text


####################################################################################

# Create a Gradio interface with a Textbox for input and a Checkbox for selecting word sets
interface1 = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.inputs.Textbox(lines=5, label="Enter your text"),
        gr.inputs.CheckboxGroup(
            label="Select Predetermined Word Sets",
            choices=list(predetermined_word_sets.keys()),  # Use the keys as checkbox labels
            type="value",  
            default=[list(predetermined_word_sets.keys())[0]],  # Set the default value / selected checkbox
        ),
    ],
    outputs=[
        gr.outputs.Textbox(label="Chatbot Response"),
        gr.outputs.Textbox(label="Scenario Analysis"),
        gr.outputs.Textbox(label="Score (%)"),
    ],
    title="Custom-trained AI Chatbot",
)


####################################################################################

interface = gr.TabbedInterface([interface1], ["User interface", "Checkbox"])

index = construct_index("docs")
interface.launch(share=True)
iface.launch(share=True, auth = ('user', 'password'), auth_message="Please log in")
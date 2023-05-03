import openai
import gradio

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a cybersecurity incident response bot"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Cyber Incident Response")

demo.launch(share=True)
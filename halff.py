import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

model_identifier = "epfl-llm/meditron-7b"

# Pass token="" to authenticate the request
model = AutoModelForCausalLM.from_pretrained(model_identifier, token="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")

# Enable mixed-precision (half-precision) inference
model.half()

model_name = "epfl-llm/meditron-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.set_page_config(page_title='DHV Meditron Chat', layout='wide')

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device).half()
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

st.title("DHV AI Startup Meditron Chat Demo")

user_input = st.text_input("User Input", value="", max_chars=100)
if st.button("Send"):
    if user_input.strip() != "":
        user_message = f"User: {user_input}"
        st.write(user_message)
        response = generate_response(user_input)
        bot_message = f"Meditron: {response}"
        st.write(bot_message)
import time, os
from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
import streamlit as st
from streamlit_chat import message

load_dotenv()
api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

def generate_response(prompt):
    user_params = GenerateParams(decoding_method="sample", max_new_tokens=100, temperature=1)
    completions = Model("ibm/falcon-40b-8lang-instruct", params=user_params, credentials=creds)

    user_response = completions.generate([prompt])
    user_gen = user_response[0].generated_text

    ##message = user_gen.replace("\n", "")

    return user_gen

st.header("IBM Generative AI App")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

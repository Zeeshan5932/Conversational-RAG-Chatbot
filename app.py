import os
import streamlit as st
import dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

st.set_page_config(page_title="Converational Chatbot", page_icon=":earth_americas:")

st.header("Hey, Lets Chat!")

dotenv.load_dotenv()

chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.5,
)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="Hello, I am a chatbot. I am here to help you with your queries. Please ask me anything!")
    ]

def get_groq_response(query):
    st.session_state['flowmessages'].append(HumanMessage(content=query))
    answer = chat.invoke(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content

input = st.text_input("Input: ", key="input")
submit = st.button("Submit")

if submit and input.strip():
    response = get_groq_response(input)
    st.write(response)
import os
import streamlit as st
import dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Converational Chatbot", page_icon=":earth_americas:")

st.header("Hey, Lets Chat!")

dotenv.load_dotenv()

chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.5,
)

search = DuckDuckGoSearchRun()

if "flowmessages" not in st.session_state:
    st.session_state["flowmessages"] = [
        SystemMessage(content="Hello, I am a chatbot. I am here to help you with your queries. Please ask me anything!")
    ]

def get_realtime_response(query):
    st.session_state["flowmessages"].append(HumanMessage(content=query))

    try:
        web_results = search.run(query)
    except Exception as exc:
        web_results = f"Live search currently unavailable: {exc}"

    system_instruction = SystemMessage(
        content=(
            "You are a helpful assistant. Use the provided live web search results to answer with"
            " up-to-date information. If the web results are missing or insufficient, clearly say"
            " that the data may be incomplete."
        )
    )
    user_with_context = HumanMessage(
        content=(
            f"User question: {query}\n\n"
            f"Live web search results:\n{web_results}\n\n"
            "Provide a clear answer based on this latest information."
        )
    )

    answer = chat.invoke([system_instruction, user_with_context])
    st.session_state["flowmessages"].append(AIMessage(content=answer.content))

    return answer.content, web_results

input = st.text_input("Ask anything (real-time web enabled):", key="input")
submit = st.button("Submit")

if submit and input.strip():
    response, web_results = get_realtime_response(input)
    st.write(response)
    with st.expander("Live search context used"):
        st.write(web_results)
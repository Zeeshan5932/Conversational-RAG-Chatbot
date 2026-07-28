import streamlit as st
import requests


def render_chat_interface(backend_url: str):
    """Renders main interactive chat stream and input interface."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "route" in msg:
                st.caption(f"⚡ Route selected: `{msg['route']}`")

    # Handle user prompt
    if prompt := st.chat_input("Ask a question, request document insights, or submit a query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking & routing..."):
                try:
                    res = requests.post(
                        f"{backend_url}/api/v1/chat",
                        json={"message": prompt},
                        timeout=45
                    )
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No response received.")
                        route = data.get("route_used", "general")
                        
                        st.markdown(answer)
                        st.caption(f"⚡ Route selected: `{route}`")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "route": route
                        })
                    else:
                        error_msg = f"API Error ({res.status_code}): {res.text}"
                        st.error(error_msg)
                except Exception as e:
                    st.error(f"Failed to communicate with agent backend: {str(e)}")
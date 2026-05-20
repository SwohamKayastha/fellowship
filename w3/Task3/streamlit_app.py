import pandas as pd
import streamlit as st

from executor import run_pipeline

st.set_page_config(page_title="Text-to-SQL", page_icon="db")

st.title("Text-to-SQL Pipeline")
st.caption("Ask a question and the system will generate and execute SQL.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the database")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and running query..."):
            response = run_pipeline(prompt)

        status = response.get("status", "failed")
        retry_count = response.get("retry_count", 0)

        st.markdown(f"Status: {status}")
        if retry_count:
            st.markdown(f"Retry needed: {retry_count}")

        st.markdown("Generated SQL:")
        st.code(response.get("sql", ""), language="sql")

        if status == "success":
            result = response.get("result", [])
            if result:
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Query returned no rows.")
        else:
            st.error(response.get("error", "Unknown error"))

    summary = f"Status: {status}\n\nSQL:\n{response.get('sql', '')}"
    st.session_state["messages"].append({"role": "assistant", "content": summary})

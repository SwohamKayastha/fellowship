import json

import pandas as pd
import streamlit as st

from graph.workflow import run_workflow

st.set_page_config(page_title="Agentic Text-to-SQL", page_icon="db")

st.title("Agentic Text-to-SQL")
st.caption("Ask a question and the agentic workflow will plan, generate SQL, execute, and summarize.")

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
        with st.spinner("Running agentic workflow..."):
            response = run_workflow(prompt)

        decomposition = response.get("decomposition") or {}
        sql = response.get("sql", "")
        status = response.get("status", "failed")
        errors = response.get("error")
        summary_text = response.get("summary", "")
        result = response.get("result", [])

        st.markdown(f"Status: {status}")
        if errors:
            st.error(errors)

        st.markdown("Planning:")
        st.json(
            {
                "question": response.get("question", prompt),
                "decomposition": decomposition,
                "sql": sql,
                "result": result,
                "summary": summary_text,
                "status": status,
                "retry_count": response.get("retry_count"),
                "duration_ms": response.get("duration_ms"),
                "error": errors,
            }
        )

        st.markdown("Generated SQL:")
        st.code(sql, language="sql")

        if result:
            df = pd.DataFrame(result)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No rows or execution failed.")

        if summary_text:
            st.markdown("Summary:")
            st.write(summary_text)

    summary = json.dumps(
        {
            "status": status,
            "sql": sql,
            "summary": summary_text,
            "status": status,
        },
        indent=2,
    )
    st.session_state["messages"].append({"role": "assistant", "content": summary})

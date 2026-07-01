"""Streamlit entry point for the LLM-based document QA app."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render the initial Streamlit application shell."""

    st.set_page_config(
        page_title="Vietnamese Document Q&A",
        page_icon="📄",
        layout="wide",
    )

    st.title("LLM-based Document Q&A with Vietnamese Support")
    st.write(
        "Upload a PDF, TXT, or DOCX document, process it into a searchable "
        "RAG index, then ask grounded questions with source citations."
    )

    with st.sidebar:
        st.header("Document")
        st.file_uploader("Upload document", type=["pdf", "txt", "docx"])
        st.button("Process document", disabled=True)
        st.button("Reset vector database", disabled=True)

        st.header("Settings")
        st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=5)
        st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.3)

    st.info("Milestone 1 scaffold is ready. Document processing will be added next.")


if __name__ == "__main__":
    main()


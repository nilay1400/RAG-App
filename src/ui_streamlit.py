import streamlit as st
from rag_chain import answer_question  
from PIL import Image
import os

st.set_page_config(page_title="RAG Paper QA", layout="wide")

# Title
st.title(" Ask Paper")

# Sidebar – show t-SNE plot if it exists
with st.sidebar:
    st.header(" t-SNE Visualization")
    #st.sidebar.write("Current dir:", os.getcwd())

    if os.path.exists("./src/tsne_visualization.png"):
        st.image(Image.open("tsne_visualization.png"), use_container_width=True)
    else:
        st.info("t-SNE plot will appear here after ingestion.")

# Input field
question = st.text_input("Ask a question about the document:", placeholder="e.g., What is the main idea of the paper?")

# Button to ask
if st.button(" Get Answer") and question.strip():
    with st.spinner("Retrieving and generating..."):
        answer, sources = answer_question(question)

    st.subheader("Answer")
    st.write(answer)

    #st.subheader(" Sources")
    #for i, source in enumerate(sources):
        #with st.expander(f"Source {i+1}: {source['source']}"):
            #st.write(source["text"])



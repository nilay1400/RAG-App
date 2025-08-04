#!/bin/bash

# Start Qdrant in background
/qdrant/qdrant &

# Wait a few seconds for Qdrant to boot
sleep 5

# Start Streamlit app
streamlit run src/ui_streamlit.py

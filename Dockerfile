# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run ingestion script before starting the UI
RUN python src/ingest.py

EXPOSE 8501
CMD ["streamlit", "run", "src/ui_streamlit.py", "--server.port=8501", "--server.enableCORS=false"]

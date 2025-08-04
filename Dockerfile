FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl unzip

# Install Qdrant
RUN curl -LO https://github.com/qdrant/qdrant/releases/download/v1.7.3/qdrant-linux-x86_64.zip && \
    unzip qdrant-linux-x86_64.zip -d /qdrant && \
    rm qdrant-linux-x86_64.zip

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY start.sh .

# Make start.sh executable
RUN chmod +x start.sh

EXPOSE 6333 8501

# Start Qdrant and Streamlit together
CMD ["./start.sh"]

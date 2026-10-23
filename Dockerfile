FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Default: run the Streamlit UI
# To run CLI instead: docker run ... python src/main.py
CMD ["streamlit", "run", "src/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

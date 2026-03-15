FROM python:3.11-slim

WORKDIR /app

# Install system deps that help with scientific Python (optional but safer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["bash", "-c", "python scripts/bootstrap.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]


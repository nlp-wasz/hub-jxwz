FROM python:3.10-slim

WORKDIR /app

COPY requirements_rag.txt .
RUN pip install --no-cache-dir -r requirements_rag.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
import sys
print(f"Python executable: {sys.executable}")
try:
    import langchain
    print(f"Langchain version: {langchain.__version__}")
    print(f"Langchain path: {langchain.__file__}")
    from langchain.chains import RetrievalQA
    print("Import RetrievalQA successful")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

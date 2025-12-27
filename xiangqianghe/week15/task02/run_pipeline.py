import os
import subprocess
import sys

# 配置路径
WORK_DIR = r"d:\BaiduSyncdisk\badou全\第15周：文档解析与DeepResearch\Week15"
TASK_DIR = os.path.join(WORK_DIR, r"Task\task02")
PDF_PATH = os.path.join(WORK_DIR, r"Week15\模型论文\2507-PaddleOCR 3.0.pdf")
MINERU_PATH = os.path.join(TASK_DIR, "run_mineru_patched.py")
RAG_SCRIPT = os.path.join(TASK_DIR, "rag_qa_ollama.py")

def run_mineru():
    print(">>> Checking Mineru tool...")
    if not os.path.exists(MINERU_PATH):
        print(f">>> Mineru script not found at {MINERU_PATH}")
        return False
    
    if not os.path.exists(PDF_PATH):
        print(f">>> PDF file not found at {PDF_PATH}")
        return False

    print(f">>> Mineru found. Starting PDF parsing...")
    print(f">>> PDF Path: {PDF_PATH}")
    print(f">>> Output Dir: {TASK_DIR}")

    # 构建命令
    cmd = [
        sys.executable,
        MINERU_PATH,
        "-p", PDF_PATH,
        "-o", TASK_DIR,
        "-m", "auto",
        "--source", "modelscope"
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
        if result.returncode == 0:
            print(">>> PDF parsing successful!")
            return True
        else:
            print(">>> PDF parsing failed.")
            return False
    except subprocess.CalledProcessError as e:
        print(f">>> Error running Mineru: {e}")
        return False
    except Exception as e:
        print(f">>> Unexpected error: {e}")
        return False

def run_rag():
    print(">>> Starting RAG QA system...")
    if not os.path.exists(RAG_SCRIPT):
        print(f">>> RAG script not found at {RAG_SCRIPT}")
        return

    query = "What are the main improvements in PaddleOCR 3.0?"
    cmd = [sys.executable, RAG_SCRIPT, query]

    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f">>> Error running RAG script: {e}")
    except Exception as e:
        print(f">>> Unexpected error: {e}")

def main():
    # 1. 运行 Mineru
    if run_mineru():
        # 2. 运行 RAG
        run_rag()
    else:
        print(">>> Skipping RAG QA due to Mineru failure.")

if __name__ == "__main__":
    main()

# Set encoding to UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Configure paths
$WorkDir = "d:\BaiduSyncdisk\badou全\第15周：文档解析与DeepResearch\Week15"
$TaskDir = Join-Path $WorkDir "Task\task02"
$PdfPath = Join-Path $WorkDir "Week15\模型论文\2507-PaddleOCR 3.0.pdf"
$MineruPath = "D:\AI\AAnaconda\Scripts\mineru.exe"

# 1. Check and run Mineru
Write-Host ">>> Checking Mineru tool..."
if (Test-Path $MineruPath) {
    Write-Host ">>> Mineru found at $MineruPath. Starting PDF parsing..."
    Write-Host ">>> PDF Path: $PdfPath"
    Write-Host ">>> Output Dir: $TaskDir"
    
    # Execute parsing
    # -p: PDF file path
    # -o: Output directory
    # -m: Mode (auto/txt/ocr), using auto
    # --source: modelscope (for faster download in China)
    & $MineruPath -p "$PdfPath" -o "$TaskDir" -m auto --source modelscope
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ">>> PDF parsing successful!"
    } else {
        Write-Host ">>> PDF parsing failed. Please check the error logs." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ">>> Mineru executable not found at $MineruPath." -ForegroundColor Yellow
    exit 1
}

# 2. Run RAG QA
$RagScript = Join-Path $TaskDir "rag_qa_ollama.py"
if (Test-Path $RagScript) {
    Write-Host ">>> Starting RAG QA system..."
    python $RagScript "What are the main improvements in PaddleOCR 3.0?"
} else {
    Write-Host ">>> RAG script not found: $RagScript" -ForegroundColor Red
}

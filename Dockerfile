FROM python:3.11-slim
WORKDIR /app
# Pin the parser stack: SymPy 1.14 requires antlr4 runtime 4.11 for parse_latex.
# Node/KaTeX supplies the primary formula gate; Chinese OCR is required for scanned papers.
RUN pip install --no-cache-dir pymupdf==1.26.5 sympy==1.14.0 antlr4-python3-runtime==4.11.* duckdb==1.4.5 rapidfuzz==3.13.0 \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm tesseract-ocr tesseract-ocr-chi-sim \
 && npm install -g katex@0.18.7 \
 && python -c "import duckdb; c=duckdb.connect(':memory:'); c.execute('INSTALL vss'); c.execute('LOAD vss')" \
 && rm -rf /var/lib/apt/lists/*
COPY paperbrain/ ./paperbrain/
COPY tools/ ./tools/
COPY demo/ ./demo/
COPY thresholds.json ./
COPY graph_policy.json ./
COPY tests/ ./tests/
EXPOSE 8000
CMD ["python", "-m", "paperbrain.server", "--port", "8000", "--host", "0.0.0.0"]

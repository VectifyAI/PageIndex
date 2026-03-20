# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Dependencias del sistema.
# DeepWiki menciona que si falla pymupdf, puede requerir libmupdf-dev. :contentReference[oaicite:1]{index=1}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala deps primero para aprovechar cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Por defecto, ejecuta el CLI (puedes pasar flags en docker run)
ENTRYPOINT ["python", "run_pageindex.py"]

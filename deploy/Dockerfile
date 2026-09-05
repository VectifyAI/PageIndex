# deploy/Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pageindex ./pageindex
RUN pip install --no-cache-dir .

# Baked in at deploy time by deploy.sh (see _book_library_data staging step);
# not present in the git tree.
COPY deploy/_book_library_data /app/book-library-data
ENV BOOKS_HOME=/app/book-library-data

EXPOSE 8080
CMD ["books", "mcp", "--transport", "http"]

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

COPY . .

RUN uv venv && uv pip install -e .

# Entrypoint et permissions
COPY entrypoint.sh wait-for-it.sh ./
RUN chmod +x entrypoint.sh wait-for-it.sh

# Définir l'interpréteur à utiliser si besoin
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["./entrypoint.sh"]

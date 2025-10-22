FROM python:3.13.5-slim-bookworm
#RUN pip install uv

#best way to install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./


RUN uv sync --locked

COPY "05-Deploying-ML-Models/predict.py" "model.bin" ./

EXPOSE 9696

ENTRYPOINT [ "uv", "run","uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]


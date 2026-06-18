# Q3B web app — container for Hugging Face Spaces (Docker SDK) or any Docker host.
FROM python:3.11-slim

# libsndfile is needed by soundfile/librosa for audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/web/requirements.txt /app/app/web/requirements.txt
RUN pip install --no-cache-dir -r /app/app/web/requirements.txt

COPY . /app

# HF Spaces routes to port 7860 by default.
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn", "--chdir", "app/web", "server:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860"]

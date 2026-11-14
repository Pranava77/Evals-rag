FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# Default: run the pipeline. Override the topic by passing args.
CMD ["python", "-m", "src.main", "Why vector databases matter"]

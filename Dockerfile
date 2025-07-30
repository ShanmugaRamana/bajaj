FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# Pre-install packages only needed for some wheels
RUN apt-get update && apt-get install -y gcc && apt-get clean

# Install only dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code after dependencies
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
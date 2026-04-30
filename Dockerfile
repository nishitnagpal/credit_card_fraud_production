# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, logic, and trained models
COPY ./app ./app
COPY ./src ./src
COPY ./models ./models

# Expose API port
EXPOSE 8000

# Run FastAPI via Uvicorn for async production performance
# Note: we use app.api:app because the file is inside the /app directory
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
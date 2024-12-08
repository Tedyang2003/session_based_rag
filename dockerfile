# Use a lightweight base image with Python 3.12
FROM python:3.12-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /rag_app

# Copy application files
COPY . /rag_app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (used by your app)
EXPOSE 5000

# Define the command to run the application
CMD ["python", "./app/main.py"]

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies (required for some image/ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY app.py .

# 7. Expose port 8000 (The default for Uvicorn)
EXPOSE 8000

# 8. Command to run the application
# Host 0.0.0.0 is CRITICAL for Docker containers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
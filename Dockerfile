# Use official slim Python base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install system dependencies (only what's needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask or any other web server
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]

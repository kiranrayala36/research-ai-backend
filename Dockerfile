# 1. Start with a lightweight, official Python image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Command to run the Ultra-fast Uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
# Use an official base image (e.g., Python, Node.js, etc.)
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Specify the command to run on container start
CMD ["python", "your_script.py"]

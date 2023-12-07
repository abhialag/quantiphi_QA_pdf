# Use the official Python image with version 3.10.9
FROM python:3.10.9

# Set the working directory in the container
WORKDIR /

# Copy the requirements.txt file into the container
# COPY requirements.txt .
# Copy the current directory contents into the container at /app
COPY . ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (if needed)
# ENV VARIABLE_NAME=value

# Specify the command to run on container start
CMD [""]
# CMD ["python", "src/pdf_ingest.py"]

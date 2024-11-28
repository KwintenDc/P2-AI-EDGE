# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the setup scripts first
RUN python scripts/save_model.py && python scripts/save_to_onnx.py

# Finally, run the continuous script
CMD ["python", "scripts/index.py"]
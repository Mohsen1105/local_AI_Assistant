# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app /app

# Install any needed packages specified in requirements.txt
# (We'll create this file next)
# RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# (Uncomment if your app is a web server)
# EXPOSE 80

# Define environment variable
# (Uncomment and set if needed)
# ENV NAME World

# Run app.py when the container launches
# (We'll create this file later)
# CMD ["python", "app.py"]

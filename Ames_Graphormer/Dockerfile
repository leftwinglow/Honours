# Use an official Miniconda image as a parent image
FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y build-essential

# Set the working directory in the container
WORKDIR /usr/src/app

COPY . .

# Update Conda
RUN conda update -n base -c defaults conda

# Install any needed packages specified in requirements.txt
RUN conda env update --name base --file /usr/src/app/linux_environment.yml

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "Graphormer_Train.py"]
FROM python:3.12-slim

# Set the working directory (you can change /app to another directory if you prefer)
WORKDIR /pyconvection

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /pyconvection

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to launch an interactive shell
CMD ["/bin/bash"]

# docker build -t pyconvection .
# docker run -it --rm --name pyconvection-dev -v "$(pwd)":/pyconvection pyconvection /bin/bash
# Dev containers: Attach to Running Container -> rust-cuda-dev
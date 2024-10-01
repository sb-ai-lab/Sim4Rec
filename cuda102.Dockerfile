# Use last torch image for our cuda
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update -y && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    wget \
    curl \
    mc \
    vim \
    nano

# Update Conda to the latest version
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -u -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda update -n base -c defaults conda

# create a Conda environment with Python 3.10 and PyTorch 1.12.1
# (last versions for cuda 10.2) 
RUN conda create -n myenv python=3.9 && \
    echo "source activate myenv" > ~/.bashrc && \
    /opt/conda/bin/conda clean -af && \
    conda install -y pytorch==1.12.1 cudatoolkit=10.2 -c pytorch -n myenv

# Activate the Conda environment
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# Install Jupyter Notebook
RUN conda install -y jupyter pandas scipy scikit-learn tqdm -n myenv

# Set up the working directory
WORKDIR /root/

# Start Bash by default when the container runs
CMD ["/bin/bash"]

FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /usr/app
COPY ./ /usr/app
RUN useradd -ms /bin/bash --uid 1000 jupyter\
 && apt-get update\
 && apt-get install -y gcc mono-mcs software-properties-common build-essential cmake openjdk-8-jdk && \
                    rm -rf /var/lib/apt/lists/* \
 && add-apt-repository -y ppa:deadsnakes/ppa\
 && apt install -y python3.9-dev python3.9-distutils curl\
 && ln -s /usr/bin/python3.9 /usr/local/bin/python3\
 && curl https://bootstrap.pypa.io/get-pip.py | python3 \
 && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
RUN apt install wget unzip
RUN pip3 install jupyterlab
RUN pip3 install jupyter
ENV JUPYTER_ENABLE_LAB=yes
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
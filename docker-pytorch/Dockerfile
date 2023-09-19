FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential\
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

## Create a non-root user and switch to it
# TODO (ivo): doe we care?
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#     && chown -R user:user /app
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

# All users can use /home/user as their home directory
# ENV HOME=/home/user
# RUN mkdir $HOME/.cache $HOME/.config \
#     && chmod -R 777 $HOME

# Set up the Conda environment (using Miniforge)
ENV PATH=/opt/mambaforge/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo /opt/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
    && chmod +x /opt/mambaforge.sh \
    && /opt/mambaforge.sh -b -p /opt/mambaforge \
    && rm /opt/mambaforge.sh \
    && mamba env update -n base -f /app/environment.yml \
    && rm /app/environment.yml \
    && mamba clean -ya

#Downloading gcloud package and Installing the package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz \
    && mkdir -p /opt/gcloud \
    && tar -C /opt/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /opt/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH="/opt/gcloud/google-cloud-sdk/bin:$PATH"

# Set the default command to python3
CMD ["python3"]

FROM jupyter/minimal-notebook

MAINTAINER David Powell <david.a.powell@anu.edu.au>

# Install gmsh and gfortran
USER root
RUN apt-get update && \
    apt-get install -y curl gfortran gmsh libxcursor1 && \
    apt-get clean

WORKDIR /opt
RUN curl -L http://geuz.org/gmsh/bin/Linux/gmsh-2.11.0-Linux64.tgz | tar zxvf -

USER jovyan

ENV GMSH_PATH /opt/gmsh-2.11.0-Linux/bin/gmsh

# Install Python 3 packages
RUN conda install --yes \
    'matplotlib=1.4*' \
    'numpy=1.10*' \
    'scipy=0.16*' \
    'setuptools=18.8*' \
    'nose=1.3*' \
    && conda clean -yt

# Install OpenModes
ADD . /opt/openmodes
WORKDIR /opt/openmodes
RUN pip install .

# Download the example notebooks
WORKDIR /tmp
RUN curl -L https://github.com/DavidPowell/openmodes-examples/archive/docker.zip -o examples.zip && \
    unzip examples.zip -d /home/jovyan/work/ && \
    mv /home/jovyan/work/openmodes-examples-docker /home/jovyan/work/examples && \
    rm examples.zip

# Trust the example notebooks    
WORKDIR /home/jovyan/work/examples
RUN mv Index.ipynb "** Start Here **.ipynb"
RUN jupyter trust *.ipynb

WORKDIR /home/jovyan/work

USER root
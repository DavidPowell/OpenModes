FROM jupyter/minimal-notebook

MAINTAINER David Powell <DavidAnthonyPowell@gmail.com>

# Install gmsh and gfortran
# Note that apt-get gmsh is required to install all dependencies
USER root
RUN apt-get update && \
    apt-get install -y curl gfortran libxcursor1 gmsh && \
    apt-get clean

ENV GMSH_VERSION 3.0.3
WORKDIR /opt
RUN curl -L http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz | tar zxvf -

USER jovyan

ENV GMSH_PATH /opt/gmsh-${GMSH_VERSION}-Linux/bin/gmsh

# Install Python 3 packages
RUN conda install --yes \
    'matplotlib=2.0*' \
    'numpy=1.13*' \
    'scipy=0.19*' \
    'setuptools=33.1*' \
    'pytest=3.1*' \
    'jinja2=2.9*' \
    'six=1.10*' \
    'ipywidgets=6.0*' \
    'dill=0.2*' \
    && conda clean -yt

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension   
    
# Install OpenModes
ADD . /opt/openmodes
WORKDIR /opt/openmodes
RUN pip install .

# Download the example notebooks
ENV EXAMPLES_VERSION 1.2.0
WORKDIR /tmp
RUN curl -L https://github.com/DavidPowell/openmodes-examples/archive/${EXAMPLES_VERSION}.zip -o examples.zip && \
    unzip examples.zip -d /home/jovyan/work/ && \
    mv /home/jovyan/work/openmodes-examples-${EXAMPLES_VERSION} /home/jovyan/work/examples && \
    rm examples.zip

# Trust the example notebooks    
WORKDIR /home/jovyan/work/examples
RUN mv Index.ipynb "** Start Here **.ipynb"
RUN jupyter trust *.ipynb

WORKDIR /home/jovyan/work
VOLUME /home/jovyan/work

USER root

CMD ["start-notebook.sh", "--NotebookApp.token=''"]

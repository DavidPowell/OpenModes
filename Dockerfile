FROM jupyter/minimal-notebook

LABEL maintainer="David Powell <DavidAnthonyPowell@gmail.com>"

# Install gmsh and gfortran
# Note that apt-get gmsh is required to install all dependencies
USER root
RUN apt-get update && \
    apt-get install -y curl gfortran libxcursor1 gmsh && \
    apt-get clean

USER jovyan

ENV JUPYTER_ENABLE_LAB yes

# Install Python 3 packages
RUN conda install --yes \
    'matplotlib=3.1.*' \
    'numpy=1.17.*' \
    'scipy=1.3.*' \
    'setuptools=42.0.*' \
    'pytest=5.3.*' \
    'jinja2=2.10.*' \
    'six=1.13.*' \
    'ipywidgets=7.5.*' \
    'dill=0.3.*' \
    'gmsh=4.4.*' \
    'meshio=3.2.*' \
    && conda clean -yt

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
    
# Install OpenModes
ADD . /opt/openmodes
WORKDIR /opt/openmodes
RUN pip install .

# Download the example notebooks
ENV EXAMPLES_VERSION 1.3.2
WORKDIR /tmp
RUN curl -L https://github.com/DavidPowell/openmodes-examples/archive/${EXAMPLES_VERSION}.zip -o examples.zip && \
    unzip -j examples.zip -d /home/jovyan/work/ && \
    rm examples.zip

# Trust the example notebooks    
WORKDIR /home/jovyan/work
RUN mv Index.ipynb "** Start Here **.ipynb"
RUN jupyter trust *.ipynb

VOLUME /home/jovyan/work

USER root

CMD ["start-notebook.sh", "--NotebookApp.token=''"]

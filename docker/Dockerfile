FROM tensorflow/tensorflow:2.9.1-gpu

# install Python package dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    rm requirements.txt

FROM tensorflow/tensorflow:2.6.0-gpu

# install Python package dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    rm requirements.txt

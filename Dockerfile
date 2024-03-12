FROM python:3.10.8-slim-bullseye

# RUN apt-get update && apt-get install -y \
#     libgdal-dev \
#     gcc

# ENV GDAL_VERSION $(gdal-config --version)


WORKDIR /prod

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY tweet_911 tweet_911
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile

CMD uvicorn tweet_911.api.fast:app --host 0.0.0.0 --port $PORT

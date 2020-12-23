FROM python:3.7

# RUN apt-get update && apt-get -y install sudo
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 \
                                        ffmpeg

# Create a working directory.
RUN mkdir /app
WORKDIR /app

COPY requirements.txt /
RUN pip install -r /requirements.txt

# RUN sudo apt install ffmpeg

COPY ./ /app
# Finally, run gunicorn. 
CMD streamlit run app.py --server.port $PORT
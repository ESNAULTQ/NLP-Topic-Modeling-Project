# python base image in the container from Docker Hub
FROM python:3.10.6-buster

# copy files to the /app folder in the container
COPY apitopic.py app/apitopic.py
COPY Pipfile app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
COPY models/ app/models/
COPY utils app/utils


# set the working directory in the container to be /app
WORKDIR /app



# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


# execute the command python main.py (in the WORKDIR) to start the app
CMD uvicorn apitopic:app --host 0.0.0.0 --port $PORT

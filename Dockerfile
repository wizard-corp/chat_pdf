###########
# BUILDER #
###########

# pull official base image
FROM python:3.10.14-slim as builder

# set work directory
WORKDIR /usr

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

RUN pip install --upgrade pip

# install dependencies
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/wheels -r requirements.txt

#########
# FINAL #
#########

# pull official base image
FROM python:3.10.14-slim

# create the app user
# RUN addgroup --system app && adduser --system --group app

# create the appropriate directories
ENV APP_HOME=/home/app
RUN mkdir -p $APP_HOME
COPY ./entrypoint.sh / $APP_HOME
COPY ./src $APP_HOME/src
RUN chmod -R 777 $APP_HOME

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd
COPY --from=builder /usr/wheels /wheels
# COPY --from=builder /usr/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

# chown all the files to the app user
# RUN chown -R app:app $APP_HOME

# change to the app user
# USER app

WORKDIR $APP_HOME

# run entrypoint.prod.sh
# ENTRYPOINT ["/home/app/entrypoint.sh"]
ENTRYPOINT ["tail", "-f", "/dev/null"]

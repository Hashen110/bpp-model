FROM python:3.10

ENV PYTHONUNBUFFERED 1

RUN mkdir /app

WORKDIR /app

RUN pip3 install pipenv

COPY ./Pipfile ./

RUN pipenv install --ignore-pipfile

COPY . ./app

CMD ["pipenv", "run", "gunicorn", "app:create_app()"]

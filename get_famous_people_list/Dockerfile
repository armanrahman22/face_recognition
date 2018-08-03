FROM python:3.6

COPY ./get_famous_people_list/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m spacy download en_core_web_lg

COPY ./get_famous_people_list/src /app/src
WORKDIR /app/src


CMD ["python", "get_famous_people_list.py"]
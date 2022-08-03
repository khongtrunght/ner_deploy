FROM python:3.9-slim

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN apt-get update && \
	apt-get install -y python3-dev  && \
	apt-get install -y gcc && \
	apt-get clean;

RUN pip install -r requirements.txt

COPY . /streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"


RUN git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git

WORKDIR /fastapi/transformers

RUN pip3 install -e .

WORKDIR /fastapi

RUN pip install -r requirements.txt


# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;



COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


import requests
import streamlit as st
from annotated_text import annotated_text


def name_entity_recognition(message):
    # annotate = get_annotation(message)
    # annotated_text(*annotate)
    url = "http://localhost:8000/ner?text="
    rsp = requests.post(url + message)
    rsp = rsp.json()
    a = [" "+i+" " if isinstance(i, str) else tuple(i) for i in rsp["text"]]
    return annotated_text(*a)
    # return a


def name_entity_recognition_lightning(message):
    rsp = requests.post("http://localhost:8000/ner_lightning?text=" + message)
    rsp = rsp.json()
    a = [" "+i+" " if isinstance(i, str) else tuple(i) for i in rsp["text"]]
    return annotated_text(*a)


def main():
    """NLP NER App"""
    st.title("COVID 19 NER Vietnamese")
    st.subheader("Let's get start")

    if st.checkbox("Normal BERT"):
        st.subheader("Named Entity Recognition")
        message = st.text_area(
            "Go Tieng viet ve covid vao day", "Type Here", key="123")
        if st.button("Start", key="start_normal"):
            # st.text_area("Result", name_entity_recognition(message))
            name_entity_recognition(message)
    if st.checkbox("CRF BERT"):
        st.subheader("Named Entity Recognition CRF")
        message = st.text_area(
            "Go Tieng viet ve covid vao day", "Type Here", key="456")
        if st.button("Start", key="start_crf"):
            # st.text_area("Result", name_entity_recognition(message))
            name_entity_recognition_lightning(message)


if __name__ == "__main__":
    main()

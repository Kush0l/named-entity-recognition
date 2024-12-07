import streamlit as st

from transformers import pipeline

checkpoint = "distilbert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
      "token-classification", model=checkpoint, aggregation_strategy="simple"
  )

st.write("enter the sentence")
x=st.text_area("enter the sentence")
is_clicked=st.button("submit")

if(is_clicked):
  st.write(token_classifier(x))

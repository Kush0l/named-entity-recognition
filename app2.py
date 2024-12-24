import streamlit as st
import pandas as pd

from transformers import pipeline

checkpoint = "distilbert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
      "token-classification", model=checkpoint, aggregation_strategy="simple"
  )

st.title("Named Entity Recognition")

x=st.text_area("Enter the sentence")
is_clicked=st.button("Find Named Entities")

if(is_clicked):
  results = token_classifier(x)

    # Convert results to a DataFrame
  df = pd.DataFrame(results)

  df["score"] = (df["score"] * 100).round(2).astype(str) + " %"

  df = df.rename(columns={
      "entity_group": "Entity Group",
      "score": "Confidence Score",
      "word": "Entity",
      "start": "Start Index",
      "end": "End Index"
  })
  
  # Display results in table format
  st.table(df)

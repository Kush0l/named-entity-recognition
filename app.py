from transformers import pipeline
import streamlit as st
import pandas as pd

from tweet import export_tweets

checkpoint = "distilbert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple", device=0  # Use device=0 for GPU, device=-1 for CPU
)


st.title("named entity recognization")

st.write("Enter some text below")

input = st.text_area("Input your text here:", height=200)

# tweet_text = export_tweets()


def get_entities():
    output = token_classifier(input)
    return output
    

if st.button("Submit"):
    entities = get_entities()
    df = pd.DataFrame(entities)
    entity_mapping = {"PER": "Person", "ORG": "Organization"}
    df["entity_group"] = df["entity_group"].map(entity_mapping)
    filtered_df = df[["entity_group", "word", "score"]]


    st.table(filtered_df)
    # for entity in entities:
    #     st.write(f"**Entity Group:** {entity['entity_group']}")
    #     st.write(f"**Word:** {entity['word']}")
    #     st.write(f"**Score:** {entity['score']:.2f}")
    #     st.write("---")
    



# print(output)


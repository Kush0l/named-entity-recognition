import time
import httpx
import pandas as pd
import streamlit as st
from transformers import pipeline
from streamlit_lottie import st_lottie
import plotly.express as px  # For interactive pie charts
import json

# Load the NER model
checkpoint = "distilbert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple"
)

# Function to fetch Reddit posts based on keyword and subreddit
def search_reddit_posts(keyword, subreddit=None, sort="relevance", limit=10):
    base_url = "https://www.reddit.com"
    url = f"{base_url}/search.json"
    dataset = []
    after_post_id = None

    headers = {"User-Agent": "RedditSearchScraper/0.1"}

    for _ in range(1):
        params = {
            "q": keyword,
            "sort": sort,
            "limit": limit,
            "after": after_post_id,
        }
        if subreddit:
            params["restrict_sr"] = 1
            params["sr_name"] = subreddit

        response = httpx.get(url, params=params, headers=headers)
        print(f'Fetching "{response.url}"...')

        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            break

        json_data = response.json()
        if "data" not in json_data or "children" not in json_data["data"]:
            print("Unexpected response structure.")
            break

        posts = [rec["data"] for rec in json_data["data"]["children"]]
        dataset.extend(posts)
        after_post_id = json_data["data"].get("after")

        if not after_post_id:  # No more pages
            print("No more posts to fetch.")
            break

        time.sleep(1)  # Respect API rate limits

    df = pd.DataFrame(dataset)
    return df

# Function to generate a table of entities and their types
def generate_entity_table(entities):
    if not entities:
        return pd.DataFrame(columns=["Entity", "Entity Type"])  # Handle empty entities

    combined_entities = []
    for entity in entities:
        word = entity["word"]
        label = entity["entity_group"]
        combined_entities.append({"Entity": word, "Entity Type": label})

    return pd.DataFrame(combined_entities)

# Function to calculate and plot entity frequency
def plot_entity_frequency(entity_table):
    if entity_table.empty:
        st.write("No entities to display in the frequency chart.")
        return

    # Count the occurrences of each entity
    entity_counts = entity_table["Entity"].value_counts()

    # Create a DataFrame for visualization
    entity_freq_df = pd.DataFrame(
        {"Entity": entity_counts.index, "Frequency": entity_counts.values}
    )

    # Plot the bar chart
    st.write("### Entity Frequency Chart:")
    st.bar_chart(entity_freq_df.set_index("Entity"))

# Function to plot a pie chart of entity types using Streamlit
def plot_entity_type_pie_chart(entity_table):
    if entity_table.empty:
        st.write("No entities to display in the pie chart.")
        return

    # Count the occurrences of each entity type
    entity_type_counts = entity_table["Entity Type"].value_counts()

    # Create a DataFrame for the pie chart
    entity_type_df = pd.DataFrame(
        {"Entity Type": entity_type_counts.index, "Count": entity_type_counts.values}
    )

    # Use Plotly for an interactive pie chart
    st.write("### Entity Type Distribution:")
    fig = px.pie(
        entity_type_df,
        names="Entity Type",
        values="Count",
    )
    st.plotly_chart(fig)

# Sidebar with animation
with st.sidebar:
    with open("animations/Animation - 1735056051984.json", "r") as f:
      animation_data = json.load(f)
    st_lottie(animation_data, key="sidebar-animation")
    
    keyword = st.text_input("Enter keyword:", value="Elon Musk")
    subreddit = st.text_input("Enter subreddit:", placeholder="optional")
    limit = st.number_input("Number of posts to fetch:", min_value=1, max_value=100, value=10)
    fetch_btn = st.button("Fetch and Analyze Reddit Posts")

# Handle Reddit post search and NER processing
if fetch_btn:
    if keyword.strip():
        # Fetch Reddit posts
        df = search_reddit_posts(keyword, subreddit=subreddit, sort="relevance", limit=limit)

        if not df.empty:
            # Process the posts with NER
            entities = []
            for _, post in df.iterrows():
                post_text = post["title"] + " " + post.get("selftext", "")
                post_entities = token_classifier(post_text)
                entities.extend(post_entities)

                # Display the post and its entities
                st.write(f"**Post Title:** {post['title']}")
               
                entity_table = generate_entity_table(post_entities)
                st.write("**Extracted Entities:**")
                st.dataframe(entity_table)
                # Add a horizontal line after each post
                st.markdown("---")

            # Plot the entity frequency graph for all entities
            st.write("## Combined Entity Analysis:")
            all_entity_table = generate_entity_table(entities)
            plot_entity_frequency(all_entity_table)
            st.markdown("---")

            # Plot the entity type pie chart
            plot_entity_type_pie_chart(all_entity_table)

        else:
            st.warning("No posts were retrieved. Try different keywords or subreddit.")

    else:
        st.warning("Please enter a keyword for Reddit search.")

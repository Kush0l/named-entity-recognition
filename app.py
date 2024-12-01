import time
import httpx
import pandas as pd
import streamlit as st
from transformers import pipeline

# Load the NER model
checkpoint = "distilbert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple"
)

# to run in mac
# token_classifier = pipeline(
#     "token-classification", model=checkpoint, aggregation_strategy="simple", device=0  # Use device=0 for GPU, device=-1 for CPU
# )

# Streamlit App Title
st.title("Reddit NER (Named Entity Recognition)")

# Input for Keyword Search
st.write("Enter a search keyword to find Reddit posts:")

# Input fields for keyword and subreddit
keyword = st.text_input("Search keyword:", value="Elon Musk")
subreddit = st.text_input("Subreddit (optional):")
sort = 'relevance'
limit = st.slider("Posts per iteration:", min_value=1, max_value=100, value=10)

# Function to fetch Reddit posts based on keyword and subreddit
def search_reddit_posts(keyword, subreddit=None, sort='relevance', limit=10):
    base_url = 'https://www.reddit.com'
    url = f"{base_url}/search.json"
    dataset = []
    after_post_id = None

    headers = {
        'User-Agent': 'RedditSearchScraper/0.1'
    }

    for _ in range(1):
        params = {
            'q': keyword,
            'sort': sort,
            'limit': limit,
            'after': after_post_id
        }
        if subreddit:
            params['restrict_sr'] = 1
            params['sr_name'] = subreddit

        response = httpx.get(url, params=params, headers=headers)
        print(f'Fetching "{response.url}"...')

        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            break

        json_data = response.json()
        if 'data' not in json_data or 'children' not in json_data['data']:
            print("Unexpected response structure.")
            break

        posts = [rec['data'] for rec in json_data['data']['children']]
        dataset.extend(posts)
        after_post_id = json_data['data'].get('after')

        if not after_post_id:  # No more pages
            print("No more posts to fetch.")
            break

        time.sleep(1)  # Respect API rate limits

    df = pd.DataFrame(dataset)
    return df

# Function to highlight entities in the text
def highlight_entities(text, entities):
    """
    Highlights entities in the text using HTML.
    """
    # Sort entities by start index (to handle overlapping properly)
    entities = sorted(entities, key=lambda x: x["start"])
    highlighted_text = ""
    last_idx = 0

    # Map entity groups to colors
    entity_colors = {
        "PER": "#ffe2f3",  # Pink
        "LOC": "#ADD8E6",  # Light Blue
        "ORG": "#90EE90",  # Light Green
        "MISC": "#FFD700"  # Gold
    }

    # Process the text and insert highlights
    for entity in entities:
        start, end, label = entity["start"], entity["end"], entity["entity_group"]
        highlighted_text += text[last_idx:start]  # Add plain text
        color = entity_colors.get(label, "#D3D3D3")  # Default to Light Gray
        highlighted_text += f'<span style="background-color: {color}; color: black; padding: 3px; border-radius: 3px; margin: 10px; ">{text[start:end]} : <strong>{label}</strong></span>'
        last_idx = end  # Update last processed index

    highlighted_text += text[last_idx:]  # Add remaining plain text
    return highlighted_text

# Function to generate a table of entities and their types
def generate_entity_table(entities):
    """
    Generates a table of entities and their types.
    """
    data = [{"Entity": entity["word"], "Entity Type": entity["entity_group"]} for entity in entities]
    df = pd.DataFrame(data)
    return df

# Handle Reddit post search and NER processing
if st.button("Fetch and Analyze Reddit Posts"):
    if keyword.strip():
        # Fetch Reddit posts
        df = search_reddit_posts(keyword, subreddit=subreddit, sort=sort, limit=limit)
        
        # Process the posts with NER
        entities = []
        highlighted_texts = []
        for _, post in df.iterrows():
            post_text = post["title"] + " " + post.get("selftext", "")
            post_entities = token_classifier(post_text)
            entities.extend(post_entities)
            highlighted_text = highlight_entities(post_text, post_entities)
            highlighted_texts.append(highlighted_text)
        
        
        
        # Display the highlighted posts
        st.write("### Reddit Posts with Named Entities Highlighted:")
        for idx, highlighted_text in enumerate(highlighted_texts):
            st.markdown(f'<p style="font-size:16px; line-height:2; margin:10px;">{highlighted_text}</p>', unsafe_allow_html=True)
    
        # Add a break line or a horizontal rule after each post for better readability
            if idx < len(highlighted_texts) - 1:  # Avoid adding a break after the last post
                st.markdown('<hr style="border: 1px solid #ccc; margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)

        
        # Generate and display the entity table
        entity_table = generate_entity_table(entities)
        st.write("### Extracted Entities and Their Types:")
        st.dataframe(entity_table)
        
    else:
        st.warning("Please enter a keyword for Reddit search.")

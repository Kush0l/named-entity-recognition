
import time 
import httpx
import pandas as pd
base_url = 'https://www.reddit.com'
endpoint = '/r/python' 
category= '/hot'
url = base_url + endpoint + category + ".json" 
after_post_id = None
dataset=[]

for _ in range(5): 
  params = {
  'limit': 100,
  't': 'year', # time unit (hour, day, week, month, year, all),
  'after': after_post_id
  }
  response = httpx.get(url, params=params)
  print (f'fetching "{response.url}"...')
  if response.status_code != 200:
    raise Exception('Failed to fetch data')
  json_data = response.json()
  # print(json_data)
  dataset.extend([rec['data'] for rec in json_data['data']['children']])
  after_post_id = json_data['data']['after'] 
  time.sleep(0.5)
df = pd.DataFrame(dataset) 

print(df)
# df.to_csv('reddit_python.csv', index=False)



'''     other approch '''

import praw

# Initialize the Reddit client
reddit = praw.Reddit(
    client_id="5jOdtVll4yCJoO5W_yTq-Q",      # Replace with your app's client ID
    client_secret="zEI9urEwG4cKHbprvLHwDV__qZyLtg",  # Replace with your app's client secret
    user_agent="MyRedditApp/0.1 (by u/Icy-Comedian-9998)"    # Example: "MyRedditApp v1.0"
)

# Define the subreddit and fetch posts
subreddit = reddit.subreddit("Python")  # Replace with your subreddit of interest

# Get top 10 hot posts
for post in subreddit.hot(limit=10):
    print(f"Title: {post.title}")
    print(f"URL: {post.url}")
    print(f"Upvotes: {post.score}")
    print("-" * 40)

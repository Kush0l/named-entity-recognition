from ntscraper import Nitter
import pandas as pd


def export_tweets():
    scraper = Nitter(0)

    tweets = scraper.get_tweets("elonmusk", mode = 'user', number=5)

    final_tweets = []
    for x in tweets['tweets']:
        data = [x['text']]
        final_tweets.append(data)


    dat = pd.DataFrame(final_tweets, columns =['text'])

    df_string = dat.to_string()

    # Join all the tweet texts into one single string
    all_text = " ".join(dat['text'].astype(str))

    print(dat)

    print(df_string)

    print(all_text)

    return all_text
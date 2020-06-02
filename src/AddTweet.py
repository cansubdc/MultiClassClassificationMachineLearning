from TwitterAccessKeys import *
import pandas as pd

url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

results = api.search(q='deprem -filter:retweets', lang='tr', count= 150)


def tweets_df(results):

        id_list = [tweet.id for tweet in results]
        data_set = pd.DataFrame(id_list, columns=["id"])
        data_set["user_name"] = [tweet.author.screen_name for tweet in results]
        data_set["text"] = [tweet.text for tweet in results]
        data_set["location"] = [tweet.author.location for tweet in results]
        data_set["created_at"] = [tweet.created_at for tweet in results]

        return data_set


df = tweets_df(results)


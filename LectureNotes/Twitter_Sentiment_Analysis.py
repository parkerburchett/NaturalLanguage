"""
Lecture number 20

source: https://www.youtube.com/watch?v=SB8ckgT8l9c&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=20

Lecture on how to query twitter source: https://pythonprogramming.net/twitter-api-streaming-tweets-python-tutorial/


This teaches how to query the twitter API.

"""
# this file does not show up on github
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from NaturalLanguage.custom_NLTK_Utils import SentimentClassifier as s
from NaturalLanguage.custom_NLTK_Utils import TwitterCreds

consumer_key = TwitterCreds.get_consumer_key()
consumer_key_secret = TwitterCreds.get_consumer_key_secret()
bearer_token = TwitterCreds.get_bearer_token()
access_token = TwitterCreds.get_access_token()
access_token_secret = TwitterCreds.get_access_token_secret()


class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data) # I just copy and pasted this code
            tweet = str(all_data["text"])
            sentimentValue = s.determine_sentiment(tweet)
            words_thatMatter = s.whatWordsMattered(tweet)

            with open("LiveTweetResults.txt","a+") as out:
                out.write("Tweet: {}\n".format(tweet))
                out.write("Classification: {}\n".format(sentimentValue))
                out.write("Based on these words: {}\n".format(words_thatMatter))
                print('Added a Tweet')
        except:
            print("Failed to add Tweet")
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_stream = Stream(auth, listener())

twitter_stream.filter(track=["movie"], languages=["en"])

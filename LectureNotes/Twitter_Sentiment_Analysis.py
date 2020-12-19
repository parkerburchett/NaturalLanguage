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
        all_data = json.loads(data) # I just copy and pasted this code
        tweet = all_data["text"]
        sentimentValue = s.determine_sentiment(tweet)
        words_thatMatter = s.whatWordsMattered(tweet)
        print("The Tweet: {} Based on these words:{}    is: {}".format(tweet,words_thatMatter, sentimentValue))
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_stream = Stream(auth, listener())

# need to filter by english and by hashtag and by number of tweets do 100 to start
twitter_stream.filter(track=["Obama"], languages=["en"])

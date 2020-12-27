"""
This is to figure out how often I classify something as Positive or Negative.

I feed in a supply of tweets in english that contains the word "hate"

I hope these are mostly negative.



"""
# this file does not show up on github
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from NaturalLanguage.custom_NLTK_Utils import Twitter_Sentiment_Classifier as s
from NaturalLanguage.custom_NLTK_Utils import TwitterCreds # you lost this somewhere need to make it again


consumer_key = TwitterCreds.get_consumer_key()
consumer_key_secret = TwitterCreds.get_consumer_key_secret()
bearer_token = TwitterCreds.get_bearer_token()
access_token = TwitterCreds.get_access_token()
access_token_secret = TwitterCreds.get_access_token_secret()

# this is copied straight from lecture


class listener(StreamListener):


    def on_data(self, data):
        try:
            all_data = json.loads(data) # I just copy and pasted this code
            tweet = str(all_data["text"])
            sentiment_value = str(s.get_sentiment(tweet))
            with open("Tweet_sentiment_resultsV5.txt","a+") as out:
                out.write("{}, {}\n".format(tweet,sentiment_value))
                print('added tweet')
        except:

            print("Failed to add Tweet")
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_stream = Stream(auth, listener())

twitter_stream.filter(track=["love"], languages=["en"])

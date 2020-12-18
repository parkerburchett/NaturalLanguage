"""
Lecture number 20

source: https://www.youtube.com/watch?v=SB8ckgT8l9c&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=20

Lecture on how to query twitter source: https://pythonprogramming.net/twitter-api-streaming-tweets-python-tutorial/


This teaches how to query the twitter API.

"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


consumer_key ="O0nXbkTwgiyKZwtptL0pZcbCg"
consumer_key_secret = "QbR8hB7TgytUlcPYNDcfC6UHydqu3aadjEADt6kuKTiFaR3A5u"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAHscKwEAAAAAfFLZaX4g%2FGPmr90VFVtMLuqY2Ws%3D2GcmIoBhQpNSMRQwCpAYQXpw0dXKb7UgE449X54WqKADHmREoL"
access_token = "872484290818228225-Ci8SAGHBvav9iyHZHi6Mo1GMkJ0DU9n"
access_token_secret = "vVu3RYfPbNbCpfplxtPVuv5AJQda5pKSXPs2zIIWguhYj"

class listener(StreamListener):

    def on_data(self, data):
        print(data)
        return True


    def on_error(self, status_code):
        print(status_code)

auth  = OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_stream = Stream(auth, listener())
twitter_stream.filter(track=["car"])






"""
simple script ot convert the kaggle dataset into something you can work with easier.
Since you only care about postive and negative sentiment and the tweet itself

"""
# for a unknown reason this reads in a 0 as '"0"'

def main():
    rawFile = open(r"training.1600000.processed.noemoticon.csv", "r")
    outFile = open(r"LabeledTweets.csv","w")
    lines = rawFile.readlines()

    for cur_line in lines:
        S =cur_line.split(",")
        score = S[0]
        tweet = S[5]

        if score == '"0"':
            toWrite = "{},{}".format(0,tweet)
            outFile.write(toWrite)

        elif score == '"4"':
            toWrite = "{},{}".format(4, tweet)
            outFile.write(toWrite)

    rawFile.close()
    outFile.close()

main()
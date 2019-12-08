from TweetSentimentAnalysis import TweetSA

tsa = TweetSA()
print(tsa.predict('I m filling very bad today...'))
print(tsa.predict('I love my life so much !!! :)'))
print(tsa.predict('I am going to school today'))
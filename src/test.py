from TweetSentimentAnalysis import TweetSA

tsa = TweetSA()

print(tsa.predict("I'm still feeling some type of way about Viserion. #GameOfThrones #crying"))
print(tsa.predict("It's a good morning today and I'm feeling lively 😊 #goodmorning #happy #lively"))
print(tsa.predict("Pound has dropped despite #UK #Govt proposals. #dropped 😤"))

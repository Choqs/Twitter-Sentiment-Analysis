# Twitter-Sentiment-Analysis

The aim of the project is to classify between -3 & 3 the negativity / positivity of a tweet. Using a large dataset of 50k samples labeled between -1 & 1, we train a model for a transfer learning. With only 3k samples of labeled data between -3 & 3, we achieve a descent accuracy.


## How to use ?

```python3
>>> from TweetSentimentAnalysis import TweetSA
Using TensorFlow backend.

>>> tsa = TweetSA()

>>> tsa.predict("I'm still feeling some type of way about Viserion. #GameOfThrones #crying #stresseating")
(-1, 'Slightly negative emotional state')

tsa.predict("It's a good morning today and I'm feeling lively 😊 #goodmorning #happy #lively")
(3, 'Very positive emotional state')

tsa.predict("Pound has dropped despite #UK #Govt proposals. It's #BECAUSE of the #arrogance of them that it's #dropped 😤")
(-3, 'Very negative emotional state')
```

## Word2Vec

Word2vecs can be found at this url:
- https://drive.google.com/file/d/10B7cvx3xN7Ef_FxwIO8sigd1J1Ibe6Lu/view
- https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

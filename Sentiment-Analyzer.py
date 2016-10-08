from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

train = [('I love this sandwich.', 'pos'),
         ('This is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('This is my best work.', 'pos'),
         ("What an awesome view", 'pos'),
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('He is my sworn enemy!', 'neg'),
         ('My boss is horrible.', 'neg'),
         ('I am sitting.', 'neu'),
         ('I am reading.', 'neu'),
         ('I own a computer.', 'neu'),
         ('This is a neutral sentence.', 'neu'),
         ('My job is to fix computers.', 'neu')]



########################################################################################################################
#
#                 Sentiment analysis using NLTK NaiveBayesClassifier
#
########################################################################################################################
train_set = [i for i,j in train]
test_set = ['Wow! this feels amazing.']

categories = ['pos', 'neg', 'neu']
stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()

all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

classifier = NaiveBayesClassifier.train(t)

test_sentence = "Wow! This feels amazing."
test_sent_features = {word.lower(): (word in word_tokenize(test_sentence.lower())) for word in all_words}

print(classifier.classify(test_sent_features))


########################################################################################################################
#
#           Sentiment Analysis of the same test case using Nearest Neighbour Classification using Cosine Similarity
#
########################################################################################################################


trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()


transformer.fit(trainVectorizerArray)

cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

for testV in testVectorizerArray:
        cos = 0.0
        ans = ''
        for n, vector in enumerate(trainVectorizerArray):
            cosine = cx(vector, testV)
            if cosine > cos:
                cos = cosine
                ans = train[n][1]

        print(ans)


transformer.fit(testVectorizerArray)

tfidf = transformer.transform(testVectorizerArray)
#DEBUG
print (tfidf.todense())
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import words
reviews = pd.read_csv('../SentimentAnalysis/reviews.csv',encoding = "ISO-8859-1")
reviews.head()

print('the total number of distinct hotels:',len(reviews.hotelname.unique())) #total number of distinct hotels
print("____")

sid = SentimentIntensityAnalyzer()
for sentence in reviews['comment'].values[:5]:
 print(sentence)
 ss = sid.polarity_scores(sentence)
 for k in sorted(ss):
  print('{0}: {1}, '.format(k, ss[k]))
 print("____")


print (reviews.groupby('hotelname')['ratings'].mean().sort_values(ascending=False).head())
print("____")

print(reviews.groupby('hotelname')['ratings'].count().sort_values(ascending=False).head())
reviews.ratings.describe()
# find the best hotels
print(reviews[(reviews.ratings >= 4.0)][['hotelname','city','ratings']].drop_duplicates().sort_values(by ='ratings',ascending = False)[:5])


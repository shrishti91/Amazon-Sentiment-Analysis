import re
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import pickle
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()


df = pd.read_csv(r"C:\Users\GAMES\Downloads\HIRSHI\SmartBridge\webpage\Amazon_cell_phone.csv")
df = df.dropna()
df = df.drop(['asin' , 'name' , 'date', 'reviewUrl', 'totalReviews', 'price', 'originalPrice', 'rating.1'], axis = 1)
a = df['rating'].tolist()
d = []
for i in range(len(a)):
    if a[i] >= 3:
        d.append(1)
    else:
        d.append(0)

df['emotion'] = d
df['Review'] = df[['title', 'body']].agg(' ' .join, axis = 1)
df = df.drop(['rating', 'comment', 'brand'] ,axis = 1)
d = []
for i in range(301):
    d.append(i)

df['index'] = d
df = df.set_index(['index'])

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(df['Review'])):

    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(corpus).toarray()
y = np.array(df.iloc[:,4].values)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

model = Sequential([
    Dense(units = 12, input_shape = (2000,), activation = 'relu'),
    Dense(units = 12, activation = 'relu'),
    #Dropout(0.3),
    Dense(units = 8, activation = 'relu'),
    #Dropout(0.3),
    #Dropout(0.5),
    Dense(units = 10, activation = 'relu'),
    #Dropout(0.5),
    Dense(units = 10, activation = 'relu'),
    Dropout(0.3),
    Dense(units = 1,activation = 'sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics =['accuracy'])
model.fit(x_train, y_train, epochs = 150, batch_size = 50)

with open("count_vec.pkl", "wb") as file:
    pickle.dump(cv, file)

with open("count_vec.pkl", "rb") as file:
    cv = pickle.load(file)

with open("amazon_review.pkl", "wb") as file:
    pickle.dump(model, file)

# pickle.dump(model, open('amazon_review.pkl', 'wb'))
# model = pickle.load(open('amazon_review.pkl', 'rb'))


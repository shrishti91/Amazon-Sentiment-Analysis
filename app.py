from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
import joblib
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = 2000)

app = Flask(__name__)

with open("amazon_review.pkl", "rb") as file:
    model = pickle.load(file)

with open("count_vec.pkl", "rb") as file:
    cv = pickle.load(file)

@app.route('/')
def hello_world():  # put application's code here
    return render_template('amazon_reviews.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    review = request.form.listvalues()

    #print(review)

    review = str(review)
    # return render_template('amazon_reviews.html', pred=review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    # all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    # return render_template('amazon_reviews.html', pred= review)

    # X = model.trasform(review)

    # y_p = model.predict(cv.transform([review]))
    y_p = model.predict(cv.transform([review]))
    # print(y_p)
    if y_p > 0.5:
        return render_template('amazon_reviews.html',pred='Positive review')
    else:
        return render_template('amazon_reviews.html',pred='Negative review')

if __name__ == '__main__':
    app.run()

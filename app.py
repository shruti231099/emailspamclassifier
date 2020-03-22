import os
from collections import Counter
import pickle

folder='email/'
files=os.listdir(folder)
emails=[folder + file for file in files]
words=[]

for email in emails:
    f=open(email, encoding='latin-1')
    blob=f.read()
    words+=blob.split(" ")
for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""
word_dict=Counter(words)
del word_dict[""]
word_dict=word_dict.most_common(3000)
features = []
labels = []
for email in emails:
    f = open(email, encoding='latin-1')
    blob = f.read().split(" ")
    data = []
    for i in word_dict:
        data.append(blob.count(i[0]))
    features.append(data)

    if 'spam' in email:
        labels.append(1)
    if 'ham' in email:
        labels.append(0)

import numpy as np
features=np.array(features)
labels=np.array(labels)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=9)

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

print(classifier.fit(X_train,y_train))
y_pred=classifier.predict(X_test)

pickle.dump(classifier,open('model.pkl','wb'))

from flask import Flask,render_template,request


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message=request.form.get('message')
    sample=[]
    for i in word_dict:
        sample.append(message.split(" ").count(i[0]))
    sample=np.array(sample)
    result=model.predict(sample.reshape(1,3000))

    if result == 0:
        return render_template('index.html',label=1)
    else:
        return render_template('index.html',label=-1)


if __name__=='__main__':
    app.run(debug=True)
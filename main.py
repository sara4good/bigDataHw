from flask import Flask
from flask import jsonify
import json
from flask import request
import pandas as pd
from hazm import *
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route('/keywordextractor', methods=['POST'])
def keyWordExtraction():
    doc=request.get_json()
    text=doc['TITLE']
    normalizer = Normalizer()
    vectorizer = TfidfVectorizer(lowercase=False, preprocessor=normalizer.normalize, tokenizer=word_tokenize,
                                 stop_words=stopwords_list())
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    denselist.sort()
    df = pd.DataFrame(denselist, columns=feature_names)
    df.sort_values(by=0, axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
    max = df.values.size
    min = int(max * 0.6)
    keyword = "";
    for i in range(min, max):
        keyword += df.keys()[i] + ","

    keyword = keyword[0:len(keyword) - 1]
    doc['keyword']=keyword
    return json.dumps(doc, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
 app.run()
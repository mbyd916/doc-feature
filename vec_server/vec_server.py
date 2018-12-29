from flask import Flask
from flask import request
from flask import jsonify
from DocSim import DocSim
import time
import numpy as np


app = Flask(__name__)
app.config.from_object('config')

ds = DocSim(app.config['WORD2VEC_MODEL_PATH'])


@app.route('/vectorize', methods=['POST'])
def vectorize():
    doc = request.form['doc'].strip()

    start = time.time()

    vector = ds.vectorize(doc)

    end = time.time()

    cost = end - start

    if len(vector) == 0:
        resp = {"code": -1, "msg": "empty vector", "cost": cost}
    else:
        resp = {"code": 0, "msg": "success", "cost": cost, "result": vector.astype(np.float32).tolist()}

    return jsonify(resp)


@app.route('/wv/<word>')
def get_word_vector(word):
    vector = ds.word_vector(word)
    if len(vector) == 0:
        return '( )'
    return '({})'.format(', '.join(map(str, vector)))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8866)

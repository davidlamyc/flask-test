import flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    args = request.args
    ibx_visits = args.get('ibx_visits')
    smart_hands_orders = args.get('smart_hands_orders')
    print(type(ibx_visits))
    print(smart_hands_orders)

    filename = 'trained_model.sav'
    # In the web page we load the file
    loaded_model = pickle.load(open(filename, 'rb'))
    # This information comes from the Web Page
    data = {
        'ibx_visits': [ibx_visits],
        'smart_hands_orders': [smart_hands_orders]
    }
    X1 = pd.DataFrame(data, columns = ['ibx_visits', 'smart_hands_orders'])
    result = loaded_model.predict(X1)
    result_as_list = result.tolist()
    print(type(result.tolist()))
    return {
        'result': result_as_list[0]
    }

app.run()
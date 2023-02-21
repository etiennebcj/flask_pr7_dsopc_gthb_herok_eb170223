from flask import Flask, jsonify, request, jsonify, render_template, url_for
import json

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
import pickle
from zipfile import ZipFile


app = Flask(__name__)

# Saved model
pickle_in = open('LGBMClassifier_best_customscore.pkl', 'rb') 
model = pickle.load(pickle_in)


# Data for modelization
z = ZipFile('train_sample_30m.zip')
sample = pd.read_csv(z.open('train_sample_30m.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
X_sample = sample.iloc[:, :-1]
data_top = X_sample.head(50)
   
    
# Predictions 
predictions_top = model.predict_proba(data_top)[:, 0]  
   

# Adding top 5 predictions on last column    
data_predictop = data_top.copy()
data_predictop['predictions'] = predictions_top
data_predictop_reset = data_predictop.reset_index()


# Sample IDs
sample_id = data_predictop_reset[['SK_ID_CURR']]


#------------------------------------------------------------------------
# Ajax home
@app.route('/')
def show_main():
	return render_template('prediction.html')


# Exemple des ID client
@app.route("/main/view ids", methods=["GET"])
def load_data():
    return sample_id.to_json(orient='values')


# Ajax predict
@app.route('/main/predict')
def show_prediction(): 
	ID = int(request.args.get('ID'))
	result = data_predictop.loc[ID:ID, 'predictions'].values[0]
	return jsonify({'Customer ID' : ID,
			'Default probability %' : (round(float(1-result)*100, 0))})


# Dataframe HTML
@app.route('/main/view dataframe', methods=['GET'])
def showData():
    	# pandas dataframe to html table flask
    	data_predictop_reset_html = data_predictop_reset.to_html()
    	return render_template('show_csv_data.html', data_var = data_predictop_reset_html)

# Data view in json
@app.route('/main/view json')
def data_json():
	return jsonify({'data top 50': data_top.to_dict()}) # top X, risque de plantage si trop de données à charger


if __name__ == "__main__":
	app.run()
	#app.run(debug=True)
    # app.run(host="localhost", port="5000", debug=True)

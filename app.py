import os
from os import environ
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle, json
import model

app = Flask(__name__)
api = Api(app)

if not os.path.isfile('classifier.pkl'):
    build_model()

with open('classifier.pkl', 'rb') as file:
	classifier = pickle.load(file)

class Classify(Resource):
	# @app.route('/predict', methods = ['POST'])
	def post(self):

		data = request.json['data']
		features = ['sepal length', 'sepal width', 'petal length', 'petal width']
		x = data["input"]

		label = {0:'setosa',1:'versicolor',2:'virginica'}
		y = classifier.predict(np.reshape(x,(1,4)))[0]

		return jsonify({'Predicted Iris class': label[y]
			})

api.add_resource(Classify,'/predict')

if __name__ == '__main__':
    # HOST = environ.get('SERVER_HOST', 'localhost')
    # PORT=8000
    # app.run(HOST,PORT,debug=True)
    app.run(debug=True)



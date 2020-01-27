import os
from os import environ
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle, json

app = Flask(__name__)
api = Api(app)

if not os.path.isfile('classifier.pkl'):
    build_model()

with open('classifier.pkl', 'rb') as file:
	classifier = pickle.load(file)

class Classify(Resource):
	# @app.route('/predict', methods = ['POST'])
	def post(self):
		self.data = request.json['data']
		self.x = [0]*4
		features = ['sepal length', 'sepal width', 'petal length', 'petal width']
		label = {0:'setosa',1:'versicolor',2:'virginica'}

		for i, feature in zip(range(4), features):
			self.x[i] = self.data[feature]
		
		# predict	
		self.y = classifier.predict(np.reshape(self.x,(1,4)))[0]
		return jsonify({'Predicted Iris class': label[self.y]
			})

api.add_resource(Classify,'/predict')
if __name__ == '__main__':
    app.run(debug=True)



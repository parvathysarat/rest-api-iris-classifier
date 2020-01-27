import sklearn
from sklearn.datasets import load_iris
from sklearn import model_selection, linear_model, metrics
import numpy as np
import pickle
from pprint import pprint

def build_model():

	data = load_iris()
	x_data, y_data = sklearn.datasets.load_iris(return_X_y=True)
	features = data['feature_names']
	# pprint(data)
	labels = {label:name for label,name in zip(range(0,3),data['target_names'])}

	# 150 samples

	# 60:40 split
	x_train,x_test,y_train,y_test=model_selection.train_test_split(x_data,y_data,test_size=0.4, random_state=10)

	# Building and fitting the model
	classifier_model = linear_model.LogisticRegression(penalty='l2', C=100, random_state = 10, multi_class = 'auto')
	classifier_model.fit(x_train,y_train)

	# Prediction
	predictions = classifier_model.predict(x_test)
	accuracy = metrics.accuracy_score(y_test, predictions)

	# Evaluation metric
	print("Accuracy of Logistic Regression model:",accuracy)

	# Printing top 2 features for each class of flower (as per classifier)
	print("\nTop features for each class:")
	for label in range(0,3):
	    top2 = np.argsort(classifier_model.coef_[label])[-2:]
	    print("%s: '%s'" % (labels[label], "', '".join(features[i] for i in top2)))
	print(metrics.classification_report(y_test, predictions))

	# Saving the trained model
	with open("classifier.pkl", 'wb') as file:
		pickle.dump(classifier_model, file)

	print(classifier_model.predict(np.reshape([0,0,0,0],(1,4))))


	print(features)
build_model()


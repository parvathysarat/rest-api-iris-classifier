# restful-api-iris-classifier

To set up required packages, run the following in the directory :

<code> pip install -r requirements.txt </code>

To run the flask application : <code> python app.py </code>

POST requests can be sent via Postman or the command line. All input features should in centimeters (cm).

Use the following json format for Postman :
```
{
	"data": 
	{
		"sepal length":3, 
		"sepal width":1.2, 
		"petal length":2.0, 
		"petal width" : 1.5

	}
} 
```

For command line : <code> curl -X POST -H "Content-type: application/json" http://127.0.0.1:5000/predict -d "{\"data\": {\"sepal length\":3, \"sepal width\":1.2, \"petal length\":2.0, \"petal width\" : 1.5}}" </code> 
(If using Windows, escape all inner quotes in the json by using /)

### Questions
1. This dataset was obviously quite small, in the product you will be working with much
more data. How would you scale your training pipeline and/or model to handle datasets
which do not easily fit into system memory?

Online or incremental learning can be used to train the model by loading the data in mini-batches that can fit in the system memory. This can be implemented in scikit-learn using partial fit where a model is fitted incrementally to more data or by other platforms like keras that allow us to specify the batch size.

2. Describe your optimal versioning strategy for APIs which expose machine learning
models. How does training the model on new data fit into versioning strategy? List the
pros and cons of your described strategy in detail.

URL versioning can be used such that the old as well as the updated endpoints are active to give users time to upgrade and the old version slowly phased out. Training on new data will have to be done while notifying clients on the features and capabilities of the new API version, at the same time not issuing any more old API keys. This will ensure that users get time to migrate and update their products.

3. Describe your choice of model and how it fits the problem. List benefits and drawbacks
of this type of model used in the way you have chosen and where there may be scaling
issues as a system like this grows in size or complexity.

I used a logistic regression multi-class model since it is a three-way classification problem with 4 features. Logistic regression can generalize well with low variance and a high bias, it is also interpretable since it can return the feature coefficients. The algorithm is known to scale well to complexity with its kernel methods that use non-linear classification boundaries for high-dimensional data and thus seems like a good choice. 

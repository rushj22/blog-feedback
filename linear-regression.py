import numpy as np
import math
import random
import itertools
import sys
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
import pylab
from sklearn.preprocessing import PolynomialFeatures
import os
from scipy.interpolate import interp1d
import csv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVC 
from sklearn.svm.classes import NuSVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


fileDir = os.path.dirname(os.path.realpath(__file__))
dataDir = os.path.join(fileDir, "BlogFeedback")


def readData(filePath):

	trainData = []

	with open(filePath, "rb") as csvFile:
		dataReader = csv.reader(csvFile, delimiter=",")

		for row in dataReader:
			trainData.append(row)
			#print row
			#break

	trainData = np.asarray(trainData, dtype=np.float)
	
	print trainData.shape
	
	rows = trainData.shape[0]
	cols = trainData.shape[1]

	trainX = trainData[:, 0:cols-1]
	trainY = trainData[:, cols-1]

	#print trainX.shape
	#print trainY.shape

	return trainX, trainY

def examineData(data):

	print data.shape
	rows = data.shape[0]
	cols = data.shape[1]

	#trainY
	labels = data[:,cols-1]

	#checking different values of labels in train set
	print set(labels)

def linearRegression():

	trainFile = os.path.join(dataDir, "blogData_train.csv")
	trainX, trainY = readData(trainFile)

	testFile = os.path.join(dataDir, "blogData_test-2012.02.09.00_00.csv")
	testX, testY = readData(testFile)


	scaler = MinMaxScaler()
	trainXScaled = scaler.fit_transform(trainX)
	testXScaled = scaler.transform(testX)

	"""
	tuned_parameters = [{'alpha': [pow(2, -6), pow(2, -5), pow(2, -3), pow(2, -1), pow(2, 1), pow(2, 3), pow(2, 5)]}]


	ridge = GridSearchCV(linear_model.Ridge(alpha=pow(2, -6)), tuned_parameters, cv=5)
	ridge.fit(trainX, trainY)
	print ridge.best_params_
	print ridge.best_score_
	"""

	"""
	tuned_parameters = [{'alpha': [pow(2, -6), pow(2, -5), pow(2, -3), pow(2, -1), pow(2, 1), pow(2, 3), pow(2, 5)]}]

	lasso = GridSearchCV(linear_model.Lasso(alpha=pow(2, -6)), tuned_parameters, cv=5)
	lasso.fit(trainXScaled, trainY)
	print lasso.best_params_
	print lasso.best_score_
	"""

	"""
	tuned_parameters = [{'alpha': [pow(2,-6), pow(2, -5), pow(2, -3), pow(2, -1), pow(2, 1), pow(2, 3), pow(2, 5)]}]

	enet = GridSearchCV(linear_model.ElasticNet(alpha=pow(2, -6), l1_ratio=0.7), tuned_parameters, cv=5)
	enet.fit(trainXScaled, trainY)
	print enet.best_params_
	print enet.best_score_
	"""
	
	"""
	lin = linear_model.LinearRegression()
	lin.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((lin.predict(testXScaled) - testY) ** 2))

	"""

	tuned_parameters = [pow(2, -6), pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1]

	ridgeScores = []
	lassoScores = []
	enetScores = []

	"""for power in tuned_parameters:

		ridge = linear_model.Ridge(alpha = power)
		scores = cross_val_score(ridge, trainXScaled, trainY, cv=5)
		score = np.mean(np.asarray(scores))
		ridgeScores.append(score)"""

	for power in tuned_parameters:

		lasso = linear_model.Lasso(alpha = power)
		scores = cross_val_score(lasso, trainXScaled, trainY, cv=5)
		score = np.mean(np.asarray(scores))
		lassoScores.append(score)

	for power in tuned_parameters:

		enet = linear_model.ElasticNet(alpha = power, l1_ratio=0.7)
		scores = cross_val_score(enet, trainXScaled, trainY, cv=5)
		score = np.mean(np.asarray(scores))
		enetScores.append(score)
	

	fig, ax = plt.subplots()
	plt.title('Lasso and ElasticNet regression hyperparameter estimation')
	#ax.plot(tuned_parameters, ridgeScores, 'r', label='Ridge')
	ax.plot(tuned_parameters, lassoScores, 'b', label='Lasso')
	ax.plot(tuned_parameters, enetScores, 'g', label='ElasticNet')
	#plt.show()
	plt.xlabel('alpha Value')
	plt.ylabel('Mean Cross Validation Score')
	legend = ax.legend(loc='upper center')
	#plt.show()
	#axes = plt.gca()
	#axes.set_xlim([xmin,xmax])
	#plt.show()
	plt.savefig("Lasso and ElasticNet Hyperparameter Estimation.png")
	plt.gcf().clear()


	"""
	ridge = linear_model.Ridge(alpha = pow(10, -5))
	ridge.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((ridge.predict(testXScaled) - testY) ** 2))

	ridgeSAG = linear_model.Ridge(alpha = pow(10, -5), solver="sparse_cg")
	ridgeSAG.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((ridgeSAG.predict(testXScaled) - testY) ** 2))
	"""

	
	"""lasso = linear_model.Lasso(alpha = 0.1)
	lasso.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((lasso.predict(testXScaled) - testY) ** 2))
	"""



	"""enet = ElasticNet(alpha=0.01, l1_ratio=0.7, solver = 'sag')
	enet.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((enet.predict(testXScaled) - testY) ** 2))
	"""

	"""mlp = MLPRegressor()
	mlp.fit(trainXScaled, trainY)
	print("Mean squared error: %.2f"
      % np.mean((mlp.predict(testXScaled) - testY) ** 2))
	"""


	"""
	sgd = SGDRegressor(penalty="l1")
	
	for c in range(3, 21):
		print c
		scores = cross_val_score(sgd, trainX, trainY, cv=int(c))
		scores1 = np.asarray(scores, dtype=np.float64)
		np.set_printoptions(suppress=True)
		np.set_printoptions(precision=3)
		print scores1
		print np.mean(scores1)
	"""
	
	"""
	mlp = MLPRegressor()
	scores = cross_val_score(mlp, trainX, trainY, cv=5)
	np.set_printoptions(suppress=True)
	print scores
	"""

	"""
	linsvc = NuSVR(C=1.0)
	scores = cross_val_score(linsvc, trainX, trainY, cv=5)
	np.set_printoptions(suppress=True)
	print scores
	"""
	#linRegr.fit()

if __name__ == "__main__":
	linearRegression()
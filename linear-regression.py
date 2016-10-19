import numpy as np
import math
import random
import itertools
import sys
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pylab
from sklearn.preprocessing import PolynomialFeatures
import os
from scipy.interpolate import interp1d
import csv

fileDir = os.path.dirname(os.path.realpath(__file__))
dataDir = os.path.join(fileDir, "BlogFeedback")





def readTrainData(filePath):

	trainData = []

	with open(filePath, "rb") as csvFile:
		dataReader = csv.reader(csvFile, delimiter=",")

		for row in dataReader:
			trainData.append(row)
			#print row
			#break

	trainData = np.asarray(trainData)
	print trainData.shape

	return trainData


def linearRegression():

	trainFile = os.path.join(dataDir, "blogData_train.csv")
	trainSet = readTrainData(trainFile)

if __name__ == "__main__":
	linearRegression()
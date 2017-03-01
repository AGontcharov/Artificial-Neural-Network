#!/usr/bin/python

import numpy as np
import sys
import csv

class NeuralNetwork():

	def __init__(self, trainingDataSet, inputNodes, hiddenNodes, outputNodes):
		self.learningRate = 0.5
		self.momentum = 0.01
		self.inputNodes = np.zeros(inputNodes)
		self.hiddenNodes = np.zeros(hiddenNodes)
		self.outputNodes = np.zeros(outputNodes)
		self.outputNodeThresholds = np.zeros(self.outputNodes.size)
		self.hiddenNodeThresholds = np.zeros(self.hiddenNodes.size)
		self.weight1Delta = np.zeros((self.inputNodes.size, self.hiddenNodes.size))
		self.weight2Delta = np.zeros((self.hiddenNodes.size, self.outputNodes.size))
		self.weight1 = np.zeros((self.inputNodes.size, self.hiddenNodes.size))
		self.weight2 = np.zeros((self.hiddenNodes.size, self.outputNodes.size))
		self.outputError = np.zeros(self.outputNodes.size)
		self.hiddenError = np.zeros(self.hiddenNodes.size)
		self.maxErrorToleration = 0.00001
		self.maxEpoch = 1000
		
		self.trainingDataSet = trainingDataSet
		self.desiredOutput = np.zeros(self.outputNodes.size)

	# Using LOG as the sigmoid function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# This is a derivate of the sigmoid function
	def sigmoidError(self, x):
		return x * (1 - x)

	def trainBP(self):

		# Initailize the weight difference for V so it has no effect on the first epoch
		for i in range(0, self.inputNodes.size):
			for j in range(0, self.hiddenNodes.size):
				self.weight1Delta[i][j] = 0.0

		# Initialize the weight difference for W so it has not effect on the first epoch
		for j in range(0, self.hiddenNodes.size):
			for k in range(0, self.outputNodes.size):
				self.weight2Delta[j][k] = 0.0

		# Generate random values for weights and thresholds
		for i in range(0, self.inputNodes.size):
			for j in range(0, self.hiddenNodes.size):
				for k in range(0, self.outputNodes.size):
					self.weight1[i][j] = np.random.uniform(0, 1)
					self.weight2[j][k] = np.random.uniform(0, 1)
					self.hiddenNodeThresholds[j] = np.random.uniform(0, 1)
					self.outputNodeThresholds[k] = np.random.uniform(0, 1)

		epoch = 0
		cumulativeError = np.inf

		while ((epoch < self.maxEpoch) and (cumulativeError > self.maxErrorToleration)):			
			cumulativeError = 0
			sampleSet = np.copy(self.trainingDataSet)

			while (sampleSet.size != 0):

				# Shuffle the sample set and pop the first row
				np.random.shuffle(sampleSet)
				pattern = sampleSet[0]

				# Divide pattern into input and output portion
				self.inputNodes = np.copy(pattern[:-3])
				self.desiredOutput = np.copy(pattern[-3:])
				sampleSet = np.delete(sampleSet, 0, axis = 0)

				# Invoke forward propagation followed by back propagation
				self.forwardPropagation()
				cumulativeError = cumulativeError + self.backPropagation()
			
			# Scale error by the number of samples
			cumulativeError = (cumulativeError / (self.trainingDataSet.shape[0] * self.outputNodes.size))
			epoch = epoch + 1

		return cumulativeError

	def forwardPropagation(self):

		# Calculate hidden nodes activation values
		sum = 0
		for i in range(0, self.hiddenNodes.size):
			# Took away the -1
			for j in range(0, self.inputNodes.size):
				sum = sum + (self.inputNodes[j] * self.weight1[j][i])

			self.hiddenNodes[i] = self.sigmoid(self.hiddenNodeThresholds[i] + sum)
			
		# Calculate output nodes activation values
		sum = 0
		for i in range(0, self.outputNodes.size):
			# Took away the -1
			for j in range(0, self.hiddenNodes.size):
				sum = sum + (self.hiddenNodes[j] * self.weight2[j][i])
			
			self.outputNodes[i] = self.sigmoid(self.outputNodeThresholds[i] + sum)

	def backPropagation(self):
		singlePatternError = 0

		for i in range(0, self.outputNodes.size):
			self.outputError[i] = self.sigmoidError(self.outputNodes[i]) * (self.desiredOutput[i] - self.outputNodes[i])
			singlePatternError = singlePatternError + ((self.desiredOutput[i] - self.outputNodes[i])**2)

		sum = 0
		for i in range(0, self.hiddenNodes.size):
			# Took away the -1
			for j in range(0, self.outputNodes.size):
				sum = sum + (self.weight2[i][j] * self.outputError[j])
			
			self.hiddenError[i] = self.sigmoidError(self.hiddenNodes[i]) * sum

		# Adjust the hidden(B) to output(C) nodes connections
		for i in range(0, self.hiddenNodes.size):
			for j in range(0, self.outputNodes.size):
				if self.momentum > 0:
					self.weight2[i][j] = self.weight2[i][j] \
					+ (self.learningRate * self.hiddenNodes[i] * self.outputError[j]) \
					+ (self.momentum * self.weight2Delta[i][j])
					self.weight2Delta[i][j] = (self.learningRate * self.hiddenNodes[i] * self.outputError[j])
				else:
					self.weight2[i][j] = self.weight2[i][j] \
					+ (self.momentum * self.hiddenNodes[i] * self.outputError[j])

		# Adjust the output node thresholds
		for i in range(0, self.outputNodes.size):
			self.outputNodeThresholds[i] = self.outputNodeThresholds[i] \
			+ (self.learningRate * self.outputError[i])

		# Adjust the input(A) to hidden(B) nodes connections
		for i in range(0, self.inputNodes.size):
			for j in range(0, self.hiddenNodes.size):
				if self.momentum > 0 :

					self.weight1[i][j] = self.weight1[i][j] \
					+ (self.learningRate * self.inputNodes[i] * self.hiddenError[j]) \
					+ (self.momentum * self.weight1Delta[i][j])
					self.weight1Delta[i][j] = self.learningRate * self.inputNodes[i] * self.hiddenError[j]
				else:
					self.weight1[i][j] = self.weight1[i][j] \
					+ (self.learningRate * self.inputNodes[i] * self.hiddenError[j])

		# Adjust the hidden node thresholds
		for i in range(0, self.hiddenNodes.size):
			self.hiddenNodeThresholds[i] = self.hiddenNodeThresholds[i] \
			+ (self.learningRate * self.hiddenError[i])

		return singlePatternError

if __name__ == "__main__":

	#trainingDataSet = np.zeros(int(sys.argv[2]))
	trainingDataSet = []

	with open(sys.argv[1], 'rb') as csvfile:
		dataReader = csv.reader(csvfile, delimiter=',')
		trainingDataSet = next(dataReader)
		for row in dataReader:
			trainingDataSet = np.vstack((trainingDataSet, row))

	inputNodes = int(sys.argv[2])
	hiddenNodes = int(sys.argv[3])
	outputNodes = int(sys.argv[4])

	#print trainingDataSet
	trainingDataSet = trainingDataSet.astype(float)

	simulation = NeuralNetwork(trainingDataSet, inputNodes, hiddenNodes, outputNodes)
	#print simulation.trainingDataSet

	simulation.trainBP()

	#Run the testing data set
	with open(sys.argv[5], 'rb') as testFile:
		dataReader = csv.reader(testFile, delimiter=',')
		for row in dataReader:
			simulation.inputNodes = np.asarray(row[:-3])
			simulation.inputNodes = simulation.inputNodes.astype(float)
			simulation.desiredOutput = np.asarray(row[-3:])
			simulation.desiredOutput = simulation.desiredOutput.astype(float)
			simulation.forwardPropagation()
 
			for values in simulation.inputNodes:
				sys.stdout.write('{0}, '.format(values))
			for values in simulation.desiredOutput:
				sys.stdout.write('{0}, '.format(values))
			for values in simulation.outputNodes[:-1]:
				sys.stdout.write('{0}, '.format(values))
			sys.stdout.write('{0}'.format(simulation.outputNodes[-1]))
			print

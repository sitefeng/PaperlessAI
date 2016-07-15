#
#  Created by Si Te Feng on Stampy Day, Jul/13/16.
#  Data queried by Dean Hillan
#  Copyright c 2016 Paperless Post. All rights reserved.
#
#

'''
This file constructs a fully connected neural network that
can predict the chance of the user selecting certain gift
card based on previous send history. The ANN is trained on
Paperless Post customer card sending data with Stochastic
Gradient Descent method with mini-batches.

The training data is SQL queried and parsed from red shift
and contains all the sent cards from all users from
 2015/01/01 to 2016-07-13
'''

import tensorflow as tf
import numpy as np
from random import randint
from DataParser import DataParser
from ArrayFormatter import ArrayFormatter


# Convenience Functions for main code
# Get the mini-batch from the full dataset and label matrices
# BatchNum starts at 0
def getBatch(batchNum, dataset, label, batchSize):
    batchStart = batchNum * batchSize
    batchEnd = batchStart + batchSize
    return (dataset[batchStart: batchEnd], label[batchStart: batchEnd])


# Cost function
def squaredErrorCost(output, target):
    diffTensor = output - target
    cost = tf.square(diffTensor)
    return cost


# Importing and parsing data
print("Reading input CSV into raw dataset...")
parser = DataParser()
rawFullDataset, packageIds = parser.readCSV(fileName="accounts_packages.csv", maxReadRows=50000)

# Shuffling the raw dataset
print("Shuffling raw input dataset...")
arrayFormatter = ArrayFormatter()
rawFullDataset = arrayFormatter.shuffle(rawFullDataset)

# Separating full datasets into processed tensors required by TensorFlow
print("Transforming parsed data into training format...")

uniquePackageCount = len(packageIds)
fullDataset, fullLabels = parser.getTrainingMatrixFromRawDataset(rawFullDataset, uniquePackageCount, 10)
# Shuffle again
fullDataset = arrayFormatter.shuffle(fullDataset)
fullLabels = arrayFormatter.shuffle(fullLabels)

print("Finished transforming data: fullDataset[%d], fullLabels[%d]" % (len(fullDataset), len(fullLabels)))


# Further separate into training and validation datasets
print("Further separating data into training and validation datasets...")

fullDatasetSize = len(fullDataset)
validationSize = int(fullDatasetSize * 0.08)

trainDataset, validDataset = parser.splitDatasetIntroTrainingAndValidation(fullDataset, validationSize)
trainLabels, validLabels = parser.splitDatasetIntroTrainingAndValidation(fullLabels, validationSize)

print("Dataset separated into training and validation portions.")
print("trainDataset[%d], trainLabels[%d] | validDataset[%d], validLabels[%d]" %
      (len(trainDataset), len(trainLabels), len(validDataset), len(validLabels)))

# Main training code with TensorFlow
print("Setting up Neural Network...")
# Constants
numTrain = len(trainDataset)
numValidation = len(validDataset)

numInput = uniquePackageCount * 2
numNodesL1 = 800
numNodesL2 = 100
numOutput = 1

batchSize = 25
stdDeviation = 0.4
learningRate = 0.7


# Temporarily reducing hyperperameter sizes for debugging
numNodesL1 = 600
numNodesL2 = 100
batchSize = 20


# Start Training
inputImgs = tf.placeholder(tf.float32, shape=[batchSize, numInput])
inputLabels = tf.placeholder(tf.float32, shape=[batchSize, numOutput])

# x.get_shape() => [batchSize, numInput, 1]
x = tf.expand_dims(inputImgs, 2)

W01 = tf.Variable(tf.random_normal([batchSize, numNodesL1, numInput], stddev=stdDeviation, dtype=tf.float32), name="weight01")
b1 = tf.Variable(tf.constant(0.01, shape=[batchSize, numNodesL1, 1]), name="bias1")

# Wx => [], b => []
z1 = tf.batch_matmul(W01, x) + b1

hidden1 = tf.nn.sigmoid(z1, name="output1")

# Second hidden layer
W12 = tf.Variable(tf.random_normal([batchSize, numNodesL2, numNodesL1], stddev=stdDeviation), name="weight12")
b2 = tf.Variable(tf.constant(0.01, shape=[batchSize, numNodesL2, 1]), name="bias2")
z2 = tf.batch_matmul(W12, hidden1) + b2

hidden2 = tf.nn.sigmoid(z2, name="output2")

# Output layer
W23 = tf.Variable(tf.random_normal([batchSize, numOutput, numNodesL2], stddev=stdDeviation), name="weight23")
# z3.get_shape() => [batchSize, outputSize, 1]
temp_z3 = tf.batch_matmul(W23, hidden2)

z3 = tf.reshape(temp_z3, [batchSize, numOutput])
y3 = tf.nn.sigmoid(z3, name="output3")

y_ = tf.reshape(inputLabels, [batchSize, numOutput])

outputErrors = squaredErrorCost(y3, y_)
avgError = tf.reduce_mean(outputErrors)

# Minimize error though backpropagation
global_step = tf.Variable(0, name='global_step')
optimizer = tf.train.AdagradOptimizer(learningRate)
trainOp = optimizer.minimize(avgError, global_step=global_step)

# Initialize Session and Variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("Training Neural Network...")
stepsToTrain = numTrain // batchSize
# Due to limited computing power, steps have been limited to 1000
stepsToTrain = stepsToTrain

batchNum = 0
for step in xrange(stepsToTrain):
    (batch_x, batch_t) = getBatch(batchNum, trainDataset, trainLabels, batchSize)

    _, currError, logit, output = sess.run([trainOp, avgError, z3, y3], feed_dict={inputImgs: batch_x, inputLabels: batch_t})
    print("TrainStep[%d/%d], Error[%f]" % (step, stepsToTrain, currError))

    batchNum += 1



# Post training validation, Classification Accuracy
outputToTargetDiff = tf.abs(tf.round(y3) - tf.round(y_))
outputToTargetNeg = tf.sub(outputToTargetDiff, 1)
outputToTargetEquality = tf.neg(outputToTargetNeg)
classificationAccuracy = tf.reduce_mean(tf.cast(outputToTargetEquality, tf.float32))

print("Validating Neural Network Accuracy...")

stepsToValidate = int(numValidation / batchSize)
accuracySum = 0
batchNum = 0
for step in xrange(stepsToValidate):
    (valid_batch_x, valid_batch_t) = getBatch(batchNum, validDataset, validLabels, batchSize)

    currAccuracy, diff, neg = sess.run([classificationAccuracy, outputToTargetDiff, outputToTargetNeg], feed_dict={inputImgs: valid_batch_x, inputLabels: valid_batch_t})
    # print("Validation CurrentAccuracy: %.02f" % currAccuracy)

    accuracySum += currAccuracy
    batchNum += 1

percentAccuracy = 100 * accuracySum / stepsToValidate
print("Validation Accuracy: [%f%%]" % percentAccuracy)


sess.close()


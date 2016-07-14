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


# Convenience Functions for main code

# Get the mini-batch from the full dataset and label matrices
# BatchNum starts at 0
def getBatch(batchNum, dataset, label, batchSize):
    batchStart = batchNum * batchSize
    batchEnd = batchStart + batchSize
    return (dataset[batchStart: batchEnd], label[batchStart: batchEnd])


# Importing and parsing data
parser = DataParser()
rawFullDataset, packageIds = parser.readCSV(fileName="accounts_packages.csv", maxReadRows=200000)


# Separating full datasets into processed tensors required by TensorFlow
uniquePackageCount = len(packageIds)

fullDataset = []
fullLabels = []

# Generating positive correlation data: ANN is expected to output favorably to
# these input data because it's what happened. See powerpoint for details.
for currUserHistory in rawFullDataset:

    for i, cardSentCount in np.ndenumerate(currUserHistory):
        currUserCard = np.zeros(uniquePackageCount, dtype=np.uint8)
        restOfUserHistory = np.array(currUserHistory, dtype=np.uint8)

        if cardSentCount != 0:
            currUserCard[i] = 1
            restOfUserHistory[i] = cardSentCount - 1

        combinedTrainItem = np.append(currUserCard, currUserHistory)
        fullDataset.append(combinedTrainItem)



# Further separate into training and validation datasets
fullDatasetSize = len(fullDataset)
validationSize = int(fullDatasetSize * 0.05)

trainDataset, validDataset = parser.splitDatasetIntroTrainingAndValidation(fullDataset, validationSize)
trainLabels, validLabels = parser.splitDatasetIntroTrainingAndValidation(fullLabels, validationSize)


# Main training code with TensorFlow
# Constants
numTrain = len(trainDataset)
numValidation = len(validDataset)

numInputW = 28
numInput = numInputW * numInputW
numNodesL1 = 2000
numNodesL2 = 500
numOutput = 1

batchSize = 50
learningRate = 0.5

# Start Training
inputImgs = tf.placeholder(tf.float32, shape=[batchSize, numInput])
inputLabels = tf.placeholder(tf.float32, shape=[batchSize, numOutput])

# x.get_shape() => [batchSize, numInput, 1]
x = tf.expand_dims(inputImgs, 2)

W01 = tf.Variable(tf.random_normal([batchSize, numNodesL1, numInput], stddev=0.4, dtype=tf.float32), name="weight01")
b1 = tf.Variable(tf.constant(0.1, shape=[batchSize, numNodesL1, 1]), name="bias1")

# Wx => [], b => []
z1 = tf.batch_matmul(W01, x) + b1

hidden1 = tf.nn.tanh(z1, name="output1")

# Second hidden layer
W12 = tf.Variable(tf.random_normal([batchSize, numNodesL2, numNodesL1], stddev=0.4), name="weight12")
b2 = tf.Variable(tf.constant(0.1, shape=[batchSize, numNodesL2, 1]), name="bias2")
z2 = tf.batch_matmul(W12, hidden1) + b2

hidden2 = tf.nn.tanh(z2, name="output2")

# Output layer
W23 = tf.Variable(tf.random_normal([batchSize, numOutput, numNodesL2], stddev=0.4), name="weight23")
# z3.get_shape() => [50, 10, 1]
_z3 = tf.batch_matmul(W23, hidden2)

z3 = tf.squeeze(_z3)
y3 = tf.nn.softmax(z3, name="output3")

y_ = tf.squeeze(inputLabels)

# crossEntropy = -tf.reduce_sum(y_ * tf.log(y3))
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(z3, y_)  # cross entropy error for training
avgError = tf.reduce_mean(crossEntropy)

##########
tf.scalar_summary(avgError.op.name, avgError)

global_step = tf.Variable(0, name='global_step')
optimizer = tf.train.GradientDescentOptimizer(learningRate)
trainOp = optimizer.minimize(avgError, global_step=global_step)

# Initialize Session and Variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Due to limited computing power, steps have been limited to 1000
# stepsToTrain = numTrain // batchSize
stepsToTrain = 1000

batchNum = 0
for step in xrange(stepsToTrain):
    (batch_x, batch_t) = getBatch(batchNum, trainDataset, train_labels, batchSize)

    _, currError = sess.run([trainOp, avgError], feed_dict={inputImgs: batch_x, inputLabels: batch_t})
    print("TrainStep[%d/%d], crossEntropy[%f]" % (step, stepsToTrain, currError))

    batchNum += 1

# Post training validation
# Classification Accuracy

# outputToTargetEquality = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
# classificationAccuracy = tf.reduce_mean(tf.cast(outputToTargetEquality, tf.float32))
#
# print("Validating Neural Network Accuracy...")
#
# stepsToValidate = numValidation // batchSize
# accuracySum = 0
# batchNum = 0
# for step in xrange(stepsToValidate):
#     (valid_batch_x, valid_batch_t) = getBatch(batchNum, validDataset, valid_labels, batchSize)
#
#     currAccuracy = sess.run(classificationAccuracy, feed_dict={inputImgs: valid_batch_x, inputLabels: valid_batch_t})
#     accuracySum += currAccuracy
#     batchNum += 1
#
# percentAccuracy = accuracySum / stepsToValidate * 100
# print("Validation Accuracy: [%f%%]" % percentAccuracy)





sess.close()


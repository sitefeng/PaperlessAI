#
#  Created by Si Te Feng on Jul/3/16.
#  Copyright c 2016 Si Te Feng. All rights reserved.
#
#  notMNIST download code referenced from Google Inc.
#  Data parsing partially referenced from Josh Bleecher Snyder
#
#  Not to be used for commercial purposes.
#

'''
This file constructs a fully connected neural network that
can recognize letters from "a" to "j", trained with Stochastic
Gradient Descent method with mini-batches.
The training data is downloaded and parsed from notMNIST
in this file.
'''


import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from PIL import Image
import glob
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from random import randint


####################################################
###################################################
###################################################

# Downloading data

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

############################################

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  print ("fileName: %s" % filename)
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

####################################################
###################################################
##################################################

# Loading Data

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

allLetterArray = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
allLetterArrayCount = 10


# param: image path to read from
# returns: image as a python list
def loadImage(imagePath):
    try:
        imFile = Image.open(imagePath)
    except IOError as err:
        print("IO error: {0}".format(err))
        return
    except:
        print("Unexpected error: {0}".format(err))
        return

    img = list(imFile.getdata())
    return img

def indexToOneHot(letterIndex):
    oneHotList = []
    for i in xrange(allLetterArrayCount):
        if letterIndex == i:
            oneHotList.append(1)
        else:
            oneHotList.append(0)
    return oneHotList

# param: letter string that should be read in
def loadLetterPath(letterFolderPath, letterIndex):

    globPath = letterFolderPath + "/*.png"
    images = []
    labels = []

    print("Loading letter [%s]" % allLetterArray[letterIndex])

    for filename in glob.glob(globPath): #assuming gif
        # print("loading image[%s]" % filename)
        image = loadImage(filename)
        if not image:
            continue

        images.append(image)

        # Load label for image
        label = indexToOneHot(letterIndex)
        labels.append(label)

    return (images, labels)


# private function
def loadAllImageFromFolder(folderName):
    allImages = []
    allLabels = []

    letterIndex = 0
    for letter in allLetterArray:
        letterPath = folderName + letter
        currImages, currLabels = loadLetterPath(letterPath, letterIndex)
        allImages.extend(currImages)
        allLabels.extend(currLabels)
        letterIndex += 1

    return (allImages, allLabels)


def loadAllTrainingData():
    return loadAllImageFromFolder("./notMNIST_large/")

def loadAllValidationData():
    return loadAllImageFromFolder("./notMNIST_small/")


def shuffleInUnison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# Dimension of train_dataset => [18724 784]
_train_dataset, _train_labels = loadAllTrainingData()

print("Training Data Loaded. [Count: %d]" % len(_train_dataset))

print("Shuffling Training Data...")
train_dataset, train_labels = shuffleInUnison(np.array(_train_dataset), np.array(_train_labels))
print("Shuffling Training Data Complete!")

_valid_dataset, _valid_labels = loadAllValidationData()
print("Validation Data Loaded. [Count: %d]" % len(_valid_dataset))

print("Shuffling Validation Data...")
valid_dataset, valid_labels = shuffleInUnison(np.array(_valid_dataset),np.array(_valid_labels))
print("Shuffling Validation Data Complete!")



################################################
################################################
################################################

# Convenience Functions for main code

# Get the mini-batch from the full dataset and label matrices
# BatchNum starts at 0
def getBatch(batchNum, dataset, label, batchSize):
    batchStart = batchNum * batchSize
    batchEnd = batchStart + batchSize
    return (dataset[batchStart: batchEnd], label[batchStart: batchEnd])


def oneHotFromLetterString(letter):
    if letter in allLetterArray:
        letterIndex = allLetterArray.index(letter)
        return indexToOneHot(letterIndex)
    else:
        raise ValueError('Letter cannot be processed')
        return None

# print the output character
# param: [10x1] one-hot representation of the output character
def letterStringFromOneHot(onehot):
    outputIndex = np.argmax(onehot)
    letterString = allLetterArray[outputIndex]
    return letterString


def letterStringFromIndex(index):
    return allLetterArray[index]


# param takes a numpy array
def convertToSingleChannelFromRGBAImage(image):
    useAlphaChannel = False
    for i in xrange(image[:, 3].size):
        pixelValue = image[i, 3]
        if pixelValue != 255:
            useAlphaChannel = True

    if useAlphaChannel:
        print("Using alpha channel on image")
        return image[:, 3]
    else:
        print("Using R channel on image")
        return image[:, 0]

################################################
################################################
################################################

# Constants
numTrain = len(train_dataset)
numValidation = len(valid_dataset)

numInputW = 28
numInput = numInputW * numInputW
numNodesL1 = 1176
numNodesL2 = 500
numOutput = 10

batchSize = 50
learningRate = 0.5



## Start Training
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

hidden2 = tf.nn.tanh(z2, name = "output2")

# Output layer
W23 = tf.Variable(tf.random_normal([batchSize, numOutput, numNodesL2], stddev=0.4), name="weight23")
# z3.get_shape() => [50, 10, 1]
_z3 = tf.batch_matmul(W23, hidden2)

z3 = tf.squeeze(_z3)
y3 = tf.nn.softmax(z3, name="output3")

y_ = tf.squeeze(inputLabels)

# crossEntropy = -tf.reduce_sum(y_ * tf.log(y3))
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(z3, y_) # cross entropy error for training
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

    (batch_x, batch_t) = getBatch(batchNum, train_dataset, train_labels, batchSize)

    _, currError = sess.run([trainOp, avgError], feed_dict={inputImgs: batch_x, inputLabels: batch_t})
    print("TrainStep[%d/%d], crossEntropy[%f]" % (step, stepsToTrain, currError))

    batchNum += 1

# Post training validation
# Classification Accuracy

outputToTargetEquality = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
classificationAccuracy = tf.reduce_mean(tf.cast(outputToTargetEquality, tf.float32))

print("Validating Neural Network Accuracy...")

stepsToValidate = numValidation//batchSize
accuracySum = 0
batchNum = 0
for step in xrange(stepsToValidate):
    (valid_batch_x, valid_batch_t) = getBatch(batchNum, valid_dataset, valid_labels, batchSize)

    currAccuracy = sess.run(classificationAccuracy, feed_dict={inputImgs: valid_batch_x, inputLabels: valid_batch_t})
    accuracySum += currAccuracy
    batchNum += 1

percentAccuracy = accuracySum / stepsToValidate * 100
print("Validation Accuracy: [%f%%]" % percentAccuracy)


##############################################
##############################################
##############################################
# Try identifying a letter image manually
# Prediction should match the following letter image
manualImagePath = "./letterA.png"

_manualImage = np.array(loadImage(manualImagePath))
manualImage = convertToSingleChannelFromRGBAImage(_manualImage)

# HACK: Getting matrix shape to match
manualImages = []
for i in xrange(batchSize):
    manualImages.append(manualImage)

_outputArray = sess.run(y3, feed_dict={inputImgs: manualImages})
outputArray = _outputArray[0]
outputLetter = letterStringFromOneHot(outputArray)

print("The predicted value is [%s]" % outputLetter)
print("Output Distribution:")
for i in xrange(outputArray.size):
    letter = letterStringFromIndex(i)
    percentage = outputArray[i] * 100
    print("%s:[%.1f%%]" % (letter, percentage))

sess.close()

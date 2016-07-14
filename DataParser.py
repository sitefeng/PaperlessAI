import csv
import numpy as np
from sets import Set
from random import randint


class DataParser:
    def __init__(self):
        pass

    # private
    def num(self, s):
        try:
            return int(s)
        except ValueError:
          return int(0)

    # Returns tuple (trainingDataset, validDataset) which are split from the original dataset
    # based on the validation data size specified in the parameter
    def splitDatasetIntroTrainingAndValidation(self, dataset, validDataSize):

        trainIndexCutoff = len(dataset) - validDataSize
        trainDataset = dataset[:trainIndexCutoff]
        validDataset = dataset[trainIndexCutoff:]

        return (trainDataset, validDataset)

    '''
    Param fileName specify the csv file to read from in the format of
                   accountId|packageId, assuming sorted by accountId!
    Returns a tuple with the following in order:
    - Matrix with each row representing send history of one user
      eg row1: 0 2 0 0 0 1 0 1 0 ... <3000>
    - An unordered unique set of packageIds that has been sentduring the dataset period

    * CSV parsing steps:
    1) Since some packages are not active, first store all the sent packageIds
    in an unordered set. Then get the length of the set.

    2) Create a new 2D matrix of width {numActivePackages}

    3) For each row from csv, add the package to the 2D matrix for the users row,
       since csv is sorted by accountId. If the accountId changed from last iteration,
       move down one row and add new package

    *Note:* Eliminating users who only sent 1 card, which is not useful for training.
            Thus, the returned array will not contain users with single card send history
    '''

    def readCSV(self, fileName="accounts_packages.csv", maxReadRows=200000):

        # listOfAccounts = []
        uniquePackages = Set([])

        f = open(fileName, 'rb')
        reader = csv.reader(f)

        # 1) Get packageIds
        print("Getting Unique PackageIds...")
        for row in reader:

            rowString = row[0]
            separatedElems = rowString.split("|")
            packageNumber = self.num(separatedElems[1])
            if packageNumber == 0:
                continue

            uniquePackages.add(packageNumber)

        uniquePackageCount = len(uniquePackages)
        print("Unique PackageId Count: %d" % uniquePackageCount)
        uniquePackageList = list(uniquePackages)

        # 2) Create new 2D matrix
        userSentHistory = []
        print("Parsing User Sent History...")

        # 3) Add user sent data to new matrix
        f = open(fileName, 'rb')
        reader = csv.reader(f)

        i = 0
        currAccountId = 0
        currPackages = np.empty(0)  # In sparse format
        userSentCardCount = 0
        for row in reader:
            rowString = row[0]
            separatedElems = rowString.split("|")
            accountId = self.num(separatedElems[0])
            packageId = self.num(separatedElems[1])
            if packageId == 0 or accountId == 0:
                continue
            # print("Parsing: AccId[%d], PackId[%d]" % (accountId, packageId))

            if accountId != currAccountId:
                # Current user must have sent 2 card or more to be considered for training data
                if userSentCardCount > 1:
                    userSentHistory.append(currPackages)
                # reset variables for next row
                currPackages = np.zeros(uniquePackageCount, dtype=np.uint8)
                currAccountId = accountId
                userSentCardCount = 0

            columnToAdd = uniquePackageList.index(packageId)
            currPackages[columnToAdd] = currPackages[columnToAdd] + 1
            userSentCardCount += 1

            i += 1
            if i % 1000 == 0:
                print(
                    "Processing data... [%.02fk out of max of %.02fk](%.0f%%)" % (
                        i / 1000.0, maxReadRows / 1000.0, 100 * i / maxReadRows))
            if i > maxReadRows:
                break

        # Adding the last user to history list as well
        if userSentCardCount > 1:
            userSentHistory.append(currPackages)

        f.close()
        print("User Sent History gathered: rows[%d], cols[%d]" % (len(userSentHistory), uniquePackageCount))
        return (userSentHistory, uniquePackageList)



    # Returns transformed dataset and labels ready for neural network consumption
    # Width of the return matrix is twice that of the input raw dataset
    def getTrainingMatrixFromRawDataset(self, rawFullDataset, uniquePackageCount, matrixMagnificationFactor = 10):
        rawFullDatasetCount = len(rawFullDataset)

        fullDataset = []
        fullLabels = []
        expectedLabel = np.array([1.0], dtype=np.float)
        notExpectedLabel = np.array([0.0], dtype=np.float)

        userIndex = 0
        for currUserHistory in rawFullDataset:

            for i, cardSentCount in np.ndenumerate(currUserHistory):

                currUserCard = np.zeros(uniquePackageCount, dtype=np.uint8)
                notCurrUserCard = np.zeros(uniquePackageCount, dtype=np.uint8)
                restOfUserHistory = np.array(currUserHistory, dtype=np.uint8)

                if cardSentCount != 0:
                    # Generating positive correlation data: ANN is expected to output favorably to
                    # these input data because it's what happened. See powerpoint for details.
                    currUserCard[i] = 1 * matrixMagnificationFactor
                    restOfUserHistory[i] = (cardSentCount - 1) * matrixMagnificationFactor

                    # For each positive correlation data, we generate one negative correlation
                    # data: ANN is expected to output 0 to these input data because the current
                    # card was not sent by the user in reality.
                    randomIndex = randint(0, uniquePackageCount - 1)
                    while randomIndex == i:
                        randomIndex = randint(0, uniquePackageCount - 1)
                    notCurrUserCard[randomIndex] = 1

                    # Add for positive correlation
                    combinedTrainItem = np.append(currUserCard, currUserHistory)
                    fullDataset.append(combinedTrainItem)
                    fullLabels.append(expectedLabel)

                    # Add for negative correlation
                    combinedNegativeTrainItem = np.append(notCurrUserCard, currUserHistory)
                    fullDataset.append(combinedNegativeTrainItem)
                    fullLabels.append(notExpectedLabel)

            userIndex += 1
            if userIndex % 50 == 0:
                print(
                "Transforming user history... [%.03fk/%.03fk]" % (userIndex / 1000.0, rawFullDatasetCount / 1000.0))

        return (fullDataset, fullLabels)
import csv
import numpy as np
from sets import Set


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

        trainIndexCutoff = len(trainDataset) - validDataSize
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
            if i % 10000 == 0:
                print(
                    "Processed: [%dk out of max of %dk](%.0f%%)" % (
                        i / 1000, maxReadRows / 1000, 100 * i / maxReadRows))
            if i > maxReadRows:
                break

        # Adding the last user to history list as well
        if userSentCardCount > 1:
            userSentHistory.append(currPackages)

        f.close()
        print("User Sent History gathered: rows[%d], cols[%d]" % (len(userSentHistory), uniquePackageCount))
        return (userSentHistory, uniquePackageList)

import csv

class DataParser:

	def __init__(self):
		pass

	def num(s):
	    try:
	        return int(s)
	    except ValueError:
	        return float(s)

	'''
	Param fileName specify the csv file to read from in the format of
	               accountId|packageId, assuming sorted by accountId!
	Returns a matrix with each row representing send history of one user
	eg row1: 0 2 0 0 0 1 0 1 0 ... <3000>

	CSV parsing steps:
	1)Since some packages are not active, first store all the sent packageIds
	in an unordered set. Then get the length of the set.

	2)Create a new 2D matrix of width {numActivePackages}

	3)For each row from csv, add the package to the 2D matrix for the users row,
	since csv is sorted by accountId. If the accountId changed from last iteration,
	move down one row and add new package
	'''

	def readCSV(self, fileName = "accounts_packages.csv"):

		# listOfAccounts = []
		listOfPackages = []


		with open(fileName, 'rb') as f:
			reader = csv.reader(f)
			index = 0
			for row in reader:
				
				rowString = row[0]
				separatedElems = rowString.split("|")
				accountNumber = int(separatedElems[0])
				packageNumber = int(separatedElems[1])

				print("acc[%d], pack[%d]" % (accountNumber, packageNumber))

				index += 1
				if index > 100:
					break





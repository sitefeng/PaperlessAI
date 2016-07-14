from DataParser import DataParser

dp = DataParser()
userSentHistory, uniquePackageList = dp.readCSV("accounts_packages.csv")

print("HISTORY")
print(userSentHistory)
print("-----------")
print(uniquePackageList)



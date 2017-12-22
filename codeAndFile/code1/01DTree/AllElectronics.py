from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree, metrics
from sklearn import preprocessing
# from sklearn.externals.six import StringIO
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'E:\PythonFile\codeAndFile\code1\01DTree\AllElectronics.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()


print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

print clf.feature_importances_

# #
# oneRowX = dummyX[0]
# newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
# #
# test_res = clf.predict(newRowX)
# #
# print test_res

#
predictedY = clf.predict(dummyX)
print predictedY
#
score = metrics.accuracy_score(dummyY, predictedY, True)
print (score)
# oneRowX = dummyX[0, :]
# oneRowX = dummyX[0, :]
# print("oneRowX: " + str(oneRowX))
#
# newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
# print("newRowX: " + str(newRowX))
#
# predictedY = clf.predict(newRowX)
# print("predictedY: " + str(predictedY))
#


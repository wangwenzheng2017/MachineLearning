#coding:utf8
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing, metrics
from sklearn import tree

#读取数据集
alls = open(r"./data/data.csv")
reader = csv.reader(alls)   #将数据存到reader
headers = reader.next()     #第一行数据，也就是头
print (headers)

featureList = [] #save feature Dict
labelList = []  #save label

for row in reader:
        #将每一行的label标记存到labelList
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row)-1):
            rowDict[headers[i]] = row[i]
        #将每一行的特征向量存到featureList
        featureList.append(rowDict)

#将数据集装换为0和1的格式
vec = DictVectorizer()
#对特征向量进行转换
dummyX = vec.fit_transform(featureList).toarray()
print (dummyX)
print (vec.get_feature_names())

#对标记进行转换
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print ("labelList:" + str(dummyX))

clf = tree.DecisionTreeClassifier(criterion='entropy')  #entropy 使用信息熵
clf = clf.fit(dummyX, dummyY)    #建模
print ("clf:\n" + str(clf))

#save clf to dot
with open("./data/all.dot","w") as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
#查看特征重要程度
print clf.feature_importances_
'''
#保存和加载模型
joblib.dump(clf, './data/ss.m')
ss = joblib.load('./data/ss.m')
print ss
'''

#获取一个实例，进行部分的修改
oneRowX = dummyX[0]
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
#开始预测
test_res = clf.predict(newRowX)
#打印预测结果
print test_res

#进行正确率评测
predictedY = clf.predict(dummyX)
print predictedY
#参数：正确的结果，测试结果，显示正确率还是个数
score = metrics.accuracy_score(dummyY, predictedY, True)
print (score)   #打印测试成绩
######################花的测试集##############################
print ("####################################################")
from sklearn.datasets import load_iris

iris = load_iris()
print iris
#分割数据0.1的测试数据，0.9的训练数据
from sklearn.cross_validation import train_test_split
train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0) #随机种子
print test_target
#建模
clf2 = tree.DecisionTreeClassifier(criterion='entropy')
clf2.fit(train_data,train_target)
#模型评测
y_pre = clf2.predict(test_data)
print ("y_pre:" + str(y_pre))
print (metrics.accuracy_score(test_target, y_pre, True))
print clf2.feature_importances_
'''
with open("data/iris.dot", "w") as f:
    tree.export_graphviz(clf, out_file=f)
'''
print clf2
print clf2.classes_
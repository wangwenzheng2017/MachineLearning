# import math
# def ComputeEuclideanDistance(x1, y1, x2, y2):
#     d = math.sqrt(math.pow((x1-x2), 2) + math.pow((y1-y2), 2))
#     return d
#
# d_ag = ComputeEuclideanDistance(3, 104, 18, 90)
#
# print d_ag

import math
def computeEuclideanDistance(x1,y1,x2,y2):
    return math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))

print(computeEuclideanDistance(3,104,18,90))
print(computeEuclideanDistance(2,100,18,90))
print(computeEuclideanDistance(1,81,18,90))
print(computeEuclideanDistance(101,10,18,90))
print(computeEuclideanDistance(99,5,18,90))
print(computeEuclideanDistance(98,2,18,90))

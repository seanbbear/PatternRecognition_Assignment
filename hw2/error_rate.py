import numpy as np
import math

def splitClass(feature,target):
    class1 = []
    class2 = []
    class3 = []
    for i in range(len(target)):
        if target[i] == 1:
            class1.append(feature[i])
        elif target[i] ==2:
            class2.append(feature[i])
        else:
            class3.append(feature[i])
    return class1,class2,class3

def preprocess():
    with open("./wine_data/wine.data","r") as f:
        data = f.read().split('\n')
        data.pop(-1) # 刪除list最後一項空白
        feature = []
        target = []
        for i in data:
            i = i.split(",")    
            float_i = []
            for item in i:
                float_item = float(item)
                float_i.append(float_item)
            feature.append(float_i[1:])
            target.append(int(i[0]))

        feature = np.array(feature)
        target = np.array(target)  
        class1,class2,class3 = splitClass(feature,target)
        return class1,class2,class3

def mean(arrays):
    return np.mean(arrays,axis = 0)

def std(arrays):
    return np.std(arrays,axis = 0)

def inverse(arrays):
    return np.linalg.inv(arrays)

def cov(arrays):
    return np.cov(transpose(arrays))

def transpose(arrays):
    return np.transpose(arrays)

def det(arrays):
    return np.linalg.det(arrays)

def log(num):
    return np.log(num)

def sqrt(num):
    return np.sqrt(num)

def exp(num):
    return math.exp(num)

def calculate_error(classA,classB):
    # print(mean(classA),mean(classB))
    # print(cov(classA),cov(classB))
    mean_distance = (transpose(mean(classB)-mean(classA)).dot(inverse((cov(classA)+cov(classB))/2)).dot(mean(classB)-mean(classA)))/8
    cov_distance = log( det((cov(classA)+cov(classB))/2) / sqrt(det(cov(classA)) * det(cov(classB))) ) /2
    # print(mean_distance,cov_distance)
    total_dis = mean_distance + cov_distance
    p_a = len(classA)/(len(classA)+len(classB))
    p_b = len(classB)/(len(classA)+len(classB))
    error = sqrt(p_a * p_b) * exp(-total_dis)
    return error
    
if __name__ == "__main__":
    class1,class2,class3 = preprocess()
    print (calculate_error(class1,class2))
    print (calculate_error(class1,class3))
    print (calculate_error(class2,class3))


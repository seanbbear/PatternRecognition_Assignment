from sklearn.model_selection import train_test_split
import numpy as np
import math

def preprocess(test_size,random_state):
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
        # print(data)
        x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state = random_state)   
        return x_train, x_test, y_train, y_test

def mean(arrays):
    return np.mean(arrays,axis = 0)

def std(arrays):
    return np.std(arrays,axis = 0)

def inverse(arrays):
    return np.linalg.inv(arrays)

def cov(arrays):
    return np.cov(arrays)

def transpose(arrays):
    return np.transpose(arrays)

def det(arrays):
    return np.linalg.det(arrays)

def log(num):
    return np.log(num)

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

def classCalculate(x, train_class,p):
    var = x - mean(train_class)
    trans_var = transpose(var)
    inv_cov = inverse(cov(transpose(train_class)))
    cov_x = cov(transpose(train_class))

    return -log(p) + var.dot(inv_cov).dot(trans_var)/2 + log(det(cov_x))/2

    
def classifier(x, train_class1, train_class2, train_class3):
    p1 = len(train_class1)/len(train_class1 + train_class2 + train_class3)
    p2 = len(train_class2)/len(train_class1 + train_class2 + train_class3)
    p3 = len(train_class3)/len(train_class1 + train_class2 + train_class3)

    predict_list = []
    for i in range(len(x)):
        class1_rate = classCalculate(x[i],train_class1,p1)
        class2_rate = classCalculate(x[i],train_class2,p2)
        class3_rate = classCalculate(x[i],train_class3,p3)
        
        if min(class1_rate, class2_rate, class3_rate) == class1_rate:
            predict_list.append(1)
        elif min(class1_rate, class2_rate, class3_rate) == class2_rate:
            predict_list.append(2)
        else:
            predict_list.append(3)
    return predict_list

def eval(predict,real):
    count = 0
    for i in range(len(predict)):
        if predict[i] == real[i]:
            count += 1
    return count/len(predict) * 100

if __name__ == "__main__":
    test_size = 0.5
    random_state = 1
    x_train, x_test, y_train, y_test = preprocess(test_size,random_state)
    train_class1, train_class2, train_class3 = splitClass(x_train,y_train)
    
    predict = classifier(x_test, train_class1, train_class2, train_class3)
    print("正確率:",eval(predict,y_test),"%")

    # for j in range(10):
    #     test_size = 0.5
    #     random_state = j
    #     x_train, x_test, y_train, y_test = preprocess(test_size,random_state)
    #     train_class1, train_class2, train_class3 = splitClass(x_train,y_train)
        
        
    #     predict = classifier(x_test, train_class1, train_class2, train_class3)
        
    #     print(random_state,eval(predict,y_test))
    

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:59:04 2019

@author: LO
"""

import math
import numpy as np
from sklearn import datasets

def all_same(items):
    return len(set(items)) == 1

def entropy(p1, n1):
    if p1 == 0 and n1 == 0:
        return 1
    elif p1 == 0 or n1 == 0:
        return 0
    pp = p1/(p1+n1)
    np = n1/(p1+n1)
    return -pp*math.log2(pp)-np*math.log2(np)

#information gain
def IG(p1, n1, p2, n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2, n1+n2)-num1/num*entropy(p1, n1)-num2/num*entropy(p2, n2)

#建01tree
def BuildTree01(target, feature):
    node = dict()
    node['data'] = np.arange(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while t < len(Tree):
        idx = Tree[t]['data']  # [0,1,2,3,.........13]
        if sum(target[idx]) == 0 and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 0
        elif sum(target[idx]) == len(idx) and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):  # for each feature
                pool = list(set(feature[idx, i]))  # 重複地拿掉
                pool.sort()
                for j in range(len(pool) - 1):  # for each threshold
                    thres = (pool[j] + pool[j + 1]) / 2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if feature[k][i] <= thres:
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1] == 1), sum(target[G1] == 0), sum(target[G2] == 1), sum(target[G2] == 0))
                    if thisIG > bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if bestIG > 0:
                Tree[t]['leaf'] = 0
                Tree[t]['selectf'] = bestf
                Tree[t]['threshold'] = bestthres
                Tree[t]['child'] = [len(Tree), len(Tree) + 1]
                node = dict()
                node['data'] = bestG1
                Tree.append(node)
                node = dict()
                node['data'] = bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf'] = 1
                if (sum(target[idx] == 1)) > (sum(target[idx] == 0)):
                    Tree[t]['decision'] = 1
                else:
                    Tree[t]['decision'] = 0

        t = t+1
    return Tree

#建12tree
def BuildTree12(target, feature):
    node = dict()
    node['data'] = np.arange(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while t < len(Tree):
        idx = Tree[t]['data']  # [0,1,2,3,.........13]
        if sum(target[idx]) == len(idx) and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 1
        elif sum(target[idx]) == len(idx)*2 and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):  # for each feature
                pool = list(set(feature[idx, i]))  # 重複地拿掉
                pool.sort()
                for j in range(len(pool) - 1):  # for each threshold
                    thres = (pool[j] + pool[j + 1]) / 2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if feature[k][i] <= thres:
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1] == 2), sum(target[G1] == 1), sum(target[G2] == 2), sum(target[G2] == 1))
                    if thisIG > bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if bestIG > 0:
                Tree[t]['leaf'] = 0
                Tree[t]['selectf'] = bestf
                Tree[t]['threshold'] = bestthres
                Tree[t]['child'] = [len(Tree), len(Tree) + 1]
                node = dict()
                node['data'] = bestG1
                Tree.append(node)
                node = dict()
                node['data'] = bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf'] = 1
                if (sum(target[idx] == 2)) > (sum(target[idx] == 1)):
                    Tree[t]['decision'] = 2
                else:
                    Tree[t]['decision'] = 1

        t = t+1
    return Tree

#
def BuildTree02(target, feature):
    node = dict()
    node['data'] = np.arange(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while t < len(Tree):
        idx = Tree[t]['data']  # [0,1,2,3,.........13]
        if sum(target[idx]) == 0 and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 0
        elif sum(target[idx]) == len(idx)*2 and all_same(target):
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):  # for each feature
                pool = list(set(feature[idx, i]))  # 重複地拿掉
                pool.sort()
                for j in range(len(pool) - 1):  # for each threshold
                    thres = (pool[j] + pool[j + 1]) / 2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if feature[k][i] <= thres:
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1] == 2), sum(target[G1] == 0), sum(target[G2] == 2), sum(target[G2] == 0))
                    if thisIG > bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if bestIG > 0:
                Tree[t]['leaf'] = 0
                Tree[t]['selectf'] = bestf
                Tree[t]['threshold'] = bestthres
                Tree[t]['child'] = [len(Tree), len(Tree) + 1]
                node = dict()
                node['data'] = bestG1
                Tree.append(node)
                node = dict()
                node['data'] = bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf'] = 1
                if (sum(target[idx] == 2)) > (sum(target[idx] == 0)):
                    Tree[t]['decision'] = 2
                else:
                    Tree[t]['decision'] = 0

        t = t+1
    return Tree

#預測會屬於哪種結果
def PredictClass(Tree,test_feature):
    now = 0
    while Tree[now]['leaf'] == 0:
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if test_feature[bestf] <= thres:
            now = Tree[now]['child'][0]  # now 變成左結點
        else:
            now = Tree[now]['child'][1]
    return Tree[now]['decision']

#載入iris檔案
iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
feature0 = iris.data[idx,:]
target0 = iris.target[idx]
fault = 0
r_01 = []
r_12 = []
r_02 = []


#第0筆到30筆
feature = []                                        # 2d array with 135 1d arrays
target = []                                        # 1d array with 135 values
for i in range(30,150):
    target.append(target0[i])
    feature.append(feature0[i])

test_feature = []
test_target = []
for i in range(0,30):
    test_target.append(target0[i])
    test_feature.append(feature0[i])

test_feature = np.array(test_feature)   # 轉為 np.ndarray
test_target = np.array(test_target)

target = np.array(target)    # 轉為 np.ndarray
feature = np.array(feature)

Tree01 = BuildTree01(target, feature)
Tree12 = BuildTree12(target, feature)
Tree02 = BuildTree02(target, feature)
    
for i in range(30):
    a = PredictClass(Tree01, test_feature[i])
    b = PredictClass(Tree12, test_feature[i])
    c = PredictClass(Tree02, test_feature[i])
    r_01.append(a)
    r_12.append(b)
    r_02.append(c)


#第30筆到60筆
feature = []                                        # 2d array with 135 1d arrays
target = []                                        # 1d array with 135 values
for i in range(0,30):
    target.append(target0[i])
    feature.append(feature0[i])

for i in range(60,150):
    target.append(target0[i])
    feature.append(feature0[i])

test_feature = []
test_target = []
for i in range(30,60):
    test_target.append(target0[i])
    test_feature.append(feature0[i])

test_feature = np.array(test_feature)   # 轉為 np.ndarray
test_target = np.array(test_target)

target = np.array(target)    # 轉為 np.ndarray
feature = np.array(feature)

Tree01 = BuildTree01(target, feature)
Tree12 = BuildTree12(target, feature)
Tree02 = BuildTree02(target, feature)
    
for i in range(30):
    a = PredictClass(Tree01, test_feature[i])
    b = PredictClass(Tree12, test_feature[i])
    c = PredictClass(Tree02, test_feature[i])
    r_01.append(a)
    r_12.append(b)
    r_02.append(c)


#第60筆到90筆
feature = []                                        # 2d array with 135 1d arrays
target = []                                        # 1d array with 135 values
for i in range(0,60):
    target.append(target0[i])
    feature.append(feature0[i])

for i in range(90,150):
    target.append(target0[i])
    feature.append(feature0[i])

test_feature = []
test_target = []
for i in range(60,90):
    test_target.append(target0[i])
    test_feature.append(feature0[i])

test_feature = np.array(test_feature)   # 轉為 np.ndarray
test_target = np.array(test_target)

target = np.array(target)    # 轉為 np.ndarray
feature = np.array(feature)

Tree01 = BuildTree01(target, feature)
Tree12 = BuildTree12(target, feature)
Tree02 = BuildTree02(target, feature)
    
for i in range(30):
    a = PredictClass(Tree01, test_feature[i])
    b = PredictClass(Tree12, test_feature[i])
    c = PredictClass(Tree02, test_feature[i])
    r_01.append(a)
    r_12.append(b)
    r_02.append(c)


#第90筆到120筆
feature = []                                        # 2d array with 135 1d arrays
target = []                                        # 1d array with 135 values
for i in range(0,90):
    target.append(target0[i])
    feature.append(feature0[i])

for i in range(120,150):
    target.append(target0[i])
    feature.append(feature0[i])

test_feature = []
test_target = []
for i in range(90,120):
    test_target.append(target0[i])
    test_feature.append(feature0[i])

test_feature = np.array(test_feature)   # 轉為 np.ndarray
test_target = np.array(test_target)

target = np.array(target)    # 轉為 np.ndarray
feature = np.array(feature)

Tree01 = BuildTree01(target, feature)
Tree12 = BuildTree12(target, feature)
Tree02 = BuildTree02(target, feature)
    
for i in range(30):
    a = PredictClass(Tree01, test_feature[i])
    b = PredictClass(Tree12, test_feature[i])
    c = PredictClass(Tree02, test_feature[i])
    r_01.append(a)
    r_12.append(b)
    r_02.append(c)


#第120筆到150筆
feature = []                                        # 2d array with 135 1d arrays
target = []                                        # 1d array with 135 values
for i in range(0,120):
    target.append(target0[i])
    feature.append(feature0[i])

test_feature = []
test_target = []
for i in range(120,150):
    test_target.append(target0[i])
    test_feature.append(feature0[i])

test_feature = np.array(test_feature)   # 轉為 np.ndarray
test_target = np.array(test_target)

target = np.array(target)    # 轉為 np.ndarray
feature = np.array(feature)

Tree01 = BuildTree01(target, feature)
Tree12 = BuildTree12(target, feature)
Tree02 = BuildTree02(target, feature)
    
for i in range(30):
    a = PredictClass(Tree01, test_feature[i])
    b = PredictClass(Tree12, test_feature[i])
    c = PredictClass(Tree02, test_feature[i])
    r_01.append(a)
    r_12.append(b)
    r_02.append(c)


#投票
result1 = []
for i in range(0,150):
    if(r_01[i] == r_02[i]):
        result1.append(r_01[i])
        
    elif(r_01[i] == r_12[i]):
        result1.append(r_01[i])
        
    elif(r_02[i] == r_12[i]):
        result1.append(r_02[i])
        
    else:
        result1.append(2)

#計算準確率    
x = 0
for i in range(0,150):
    if(result1[i] == target0[i]):
        x = x+1
print('預測準確率:',x/150)

#混淆矩陣
from sklearn.metrics import confusion_matrix
y_true = target0
y_pred = result1
print('confusion matrix:\n',confusion_matrix(y_true, y_pred))

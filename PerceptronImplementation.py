#!/usr/bin/env python
# coding: utf-8

# # Perceptron

# In[3]:


#library
import numpy as np


# Data:
# - Three class types
# - Numeric features
# - Training three different perceptron to distinguish between each of the three classes

# In[4]:


#Getting training data
with open('train.data', 'r') as file: 
    train = file.read().split('\n')

#Getting test data
with open('test.data', 'r') as file: 
    test = file.read().split('\n')

#Splitting each object to produce a nested list
train = [obj.split(',') for obj in train]
test = [obj.split(',') for obj in test]

#remove empty value
train = train[:-1]
test = test[:-1]

#insert ones into train
for i in train:
    i.insert(0,1)

#inserting ones into test
for i in test:
    i.insert(0,1)

#make np array
train = np.array(train)
test = np.array(test)

#copying train data
train1 = train.copy()
train2 = train.copy()

#three datasets for each class comparison
train1 = train1[:80]
train2 = train2[40:]
train3 = np.delete(train, [i for i in range(40, 81)], 0)

#class 1 = 1, class 2 = -1
train1[:40,-1:] = '1'
train1[40:,-1:] = '-1'
#class 2 = 1, class 3 = -1
train2[:40,-1:] = '1'
train2[40:,-1:] = '-1'
#class 1 = 1, class 3 = -1
train3[:40,-1:] = '1'
train3[40:,-1:] = '-1'

#floats
train1 = train1.astype(float)
train2 = train2.astype(float)
train3 = train3.astype(float)


# Training Function
# - inputs: training data (labels, features), iterations of training
# - output: weights found

# In[5]:


#training function for perceptron
def perceptronTrain(data, iterations = 20):
    #get all features and labels
    features = [i[:-1] for i in data]
    labels = [i[-1:] for i in data]
    #initialize weights and bias
    weights = [0 for i in range(len(features[0]))]    # Extra 0 to represent bias as weight 0
    #training iters
    for i in range(iterations):
        for x, label in zip(features, labels):
            #compute activation, bias inside vectors
            a = np.dot(weights, x)
            #updating weights and bias
            if label[0] * a <= 0:
                #print("\nprior weights: ", weights)
                weights = weights + (label[0] * x)
                #print("\nNew weights: ", weights)
    #return final weights      
    return weights

#train between class 1 and class 2
w1 = perceptronTrain(train1)
#train between class 2 and class 3
w2 = perceptronTrain(train2)
#train between class 1 and class 3
w3 = perceptronTrain(train3)

#printing final weights with bias as first element in lists
print(f"Weights for class 1 v.s. class 2: {w1}\nWeights for class 2 v.s. class 3: {w2}\nWeights for class 1 v.s. class 3: {w3}\n")


# Test function
# - inputs: features, weights
# - output: 1 or 0 (1 if predicts it is an instance of the class corresponding to the weights passed in, 0 if not class)

# In[6]:


#function to predict label of features passed in, using weights passed in
def p_predict(weights, features):
    a = np.dot(weights, features)
    if a > 0:
        return 1
    return 0


# Evaluating Accuracy of each classifier

# In[7]:


#Getting features of objects used for prediction
f1 = test[5,:-1].astype(float)
f2 = test[17,:-1].astype(float)
f3 = test[26,:-1].astype(float)


# In[8]:


#Predicting according to weights
print(f"Actual Class: {test[5,-1:]}, Predicted: {p_predict(w1, f1)}")
print(f"Actual Class: {test[17,-1:]}, Predicted: {p_predict(w2, f2)}")
print(f"Actual Class: {test[26,-1:]}, Predicted: {p_predict(w3, f3)}")


# In[9]:


#Convert test labels, then test to float
test[:10,-1:] = '1'
test[10:20,-1:] = '2'
test[20:,-1:] = '3'
test = test.astype(float)


# In[10]:


#Predicting class 1 test data
for features in test[:10, :-1]:
    print(p_predict(w1, features))


# In[11]:


#Predicting class 2 with class 1 classifier
for features in test[10:20, :-1]:
    print(p_predict(w1, features))


# - with first weights, all class 1 predicted correctly, all class 2 non-predicted correctly also
# - 100% accuracy

# In[12]:


#Predicting class 2 with class 2 classifier
for features in test[10:20, :-1]:
    print(p_predict(w2, features))


# In[13]:


#Predicting class 2 test data
for features in test[20:, :-1]:
    print(p_predict(w2, features))


# - difficulty seperating class 2 and class 3

# In[14]:


#Predicting class 3 test data
for features in test[:10, :-1]:
    print(p_predict(w3, features))


# In[15]:


#Predicting class 3 test data
for features in test[20:, :-1]:
    print(p_predict(w3, features))


# - all class one instances predicted as class 1 when discriminating between class 1 and class 3
# - all class 3 instances predicted as not class 1 with same weights
# - 100% classification accuracy

# One-vs-Rest Approach
# - for each class i, create a classifier where i is positive samples and all other classes are negative
# - need a new data set with all three classes

# In[16]:


#getting data set
#class 1
ovr_train1 = train.copy()
ovr_train1[:40,-1:] = '1'
ovr_train1[40:80,-1:] = '-1'
ovr_train1[80:,-1:] = '-1'

#class 2
ovr_train2 = train.copy()
ovr_train2[:40,-1:] = '-1'
ovr_train2[40:80,-1:] = '1'
ovr_train2[80:,-1:] = '-1'

#class 3
ovr_train3 = train.copy()
ovr_train3[:40,-1:] = '-1'
ovr_train3[40:80,-1:] = '-1'
ovr_train3[80:,-1:] = '1'


# In[17]:


#one-vs-rest for class 1 weights
ovr_w1 = perceptronTrain(ovr_train1.astype(float))

#one-vs-rest for class 2 weights
ovr_w2 = perceptronTrain(ovr_train2.astype(float))

#one-vs-rest for class 3 weights
ovr_w3 = perceptronTrain(ovr_train3.astype(float))


# In[ ]:


#make into ensemble of predictions
predictions = []
for features in test[:,:-1]:
    temp_predicts = []
    temp_predicts.append(p_predict(ovr_w1, features))
    temp_predicts.append(p_predict(ovr_w2, features))
    temp_predicts.append(p_predict(ovr_w3, features))
    predictions.append(temp_predicts)
print(predictions)


# L2 Regularisation

# In[18]:


#returns sigmoid of value passed in
def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[19]:


#training function for perceptron with l2 regularisation
def l2reg_perceptronTrain(data, iterations = 20, lamb=0.1, learning_rate = 1):
    #get all features and labels
    features = [i[:-1] for i in data]
    labels = [i[-1:] for i in data]
    #initialize weights and bias
    weights = [0 for i in range(len(features[0]))]    # Extra 0 to represent bias as weight 0
    weights = np.array(weights)
    #training iters
    for i in range(iterations):
        for x, label in zip(features, labels):
            #compute activation, bias inside vectors
            a = np.dot(weights, x)
            #updating weights and bias
            if label[0] * a <= 0:
                #updating weights
                weights = (1-2*learning_rate*lamb)*weights + learning_rate*label[0]*sigmoid(-label[0]*a)*x
    #return final weights      
    return weights


# In[20]:


#one-vs-rest for class 2 weights with l2 reg
reg_weights = []
coeff = [0.01, 0.1, 1, 10, 100]
for co in coeff:
    reg_weights.append(l2reg_perceptronTrain(ovr_train2.astype(float), lamb = co))
print(reg_weights)


# In[ ]:





#Machine Learning Problem Set 3
#Hannah Lyon

from sklearn import svm 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

def main():
    #Run function to plot the points in problem 1
    #problem_1()
    
    training_data, test_data, training_labels, test_labels = mnist_data()
    test = model(training_data, test_data, training_labels, test_labels)
  
    print('Test error: {}'.format(1-test))
    

#Plot the training points
def problem_1():
    x1 = (2,4,4,0,2,0)
    x2 = (2,4,0,0,0,2)
    plt.scatter(x1, x2, color = ['red','red','red','blue','blue','blue'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    return
    

#Problem 5 imports data for handwritten characters and normalizes it
def mnist_data():
    train = open('mnist_train.txt', 'r')
    test = open('mnist_test.txt', 'r')
    training_data = train.readlines()
    test_data = test.readlines()
    
    #get feature vectors of test and training data
    for e in range(len(training_data)):
        training_data[e] = training_data[e].split(',')
    for t in range(len(test_data)):
        test_data[t] = test_data[t].split(',')
    
    #get label arrays
    training_labels = []
    for k in range(len(training_data)):
        temp = training_data[k].pop(0)
        training_labels.append(temp)
    
    test_labels = []
    for m in range(len(test_data)):
        temp = test_data[m].pop(0)
        test_labels.append(temp)
    
    #convert to arrays
    for z in range(len(training_data)):
        training_data[z] = np.array(training_data[z], dtype = float)
        training_labels[z] = float(training_labels[z])
    for x in range(len(test_data)):
        test_data[x] = np.array(test_data[x], dtype = float)
        test_labels[x] = float(test_labels[x])
        
    training_data = np.array(training_data)
    test_data = np.array(test_data)
    
    #normalize the data
    for i in range(len(training_data)):
        training_data[i] = ((2 * training_data[i]) / 255) - 1
    for j in range(len(test_data)):
        test_data[j] = ((2 * test_data[j]) / 255) - 1
        
    return training_data, test_data, training_labels, test_labels
    

#This function uses the Gaussian kernel to implement the support vector machine
def model(training_data, test_data, training_labels, test_labels):
    model = svm.SVC(gamma = 0.005, C=1, kernel = 'rbf')
    model.fit(training_data,training_labels)
    accuracy = model.score(test_data, test_labels)
    
    #5 fold cross validation scores
    scores = cross_val_score(model,training_data, training_labels, cv=5)
    print("Cross Validation Scores: {}".format(scores))
    
    
    #Test combinations of different parameter values
    #gamma = [0.005, 0.01, 0.1, 0.5, 1]
    #c = [0.01, 1, 3, 5, 10]
    
    #for penalty in c:
        #for g in gamma:
            #model = svm.SVC(C = penalty, gamma = g)
            #model.fit(training_data, training_labels)
            #scores = cross_val_score(model,training_data, training_labels, cv=5)
            #print("Cross Validation Scores given gamma = {} and C = {} : {}".format(g, penalty, scores))

    return accuracy 

main()

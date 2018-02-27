import csv
import numpy as np
import pandas as pd
from math import sqrt,pi
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from numpy import array
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from comtypes.safearray import numpy
from sklearn.linear_model.base import LinearRegression
from collections import OrderedDict




Y = []
X = []


with open('hw1-x.csv', 'rb') as f:
    reader_x = csv.reader(f, delimiter = ' ')
    
    for row in reader_x:
        row = map(float, row)
        X.append(row)
    
with open('hw1-y.csv', 'rb') as f:
    reader_y = csv.reader(f, delimiter = '\n')
 
    for row in reader_y:
        row = map(float, row)
        Y.append(row)
 


 

 


X = array(X)
Y = array(Y)

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)


clf = Ridge()
#clf.fit(X_train, y_train)
alpha_ridge = [0,0.1,1,10,100,1000,10000,100000]
errors_test = []
errors_train = []
L2Norm_weights = []

weights = []


for i in alpha_ridge:
    clf.set_params(alpha=i)
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    
    errors_test.append( sqrt(mean_squared_error(y_test,Y_pred)) )
    
    Y_pred2 = clf.predict(X_train)
    errors_train.append(sqrt(mean_squared_error(y_train,Y_pred2))) 
    L2Norm_weights.append((np.linalg.norm(clf.coef_)))
    weights.append(clf.coef_)




weights = array(weights)


weights = np.matrix.transpose(weights)

 
#************************ PART (A) AND (B)************************************
plt.subplot(311)
plt.plot(alpha_ridge,errors_test)
plt.plot(alpha_ridge,errors_train)

plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')


plt.subplot(312)
plt.plot(alpha_ridge,L2Norm_weights)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('L2Norm_weights')
plt.title('L2 Norm of Weights as a function of the regularization')


plt.subplot(313)



for j in weights:
    plt.plot(alpha_ridge, np.ndarray.tolist(j)[0])
 
 
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('weights as a function of the regularization')

plt.show()
plt.savefig('File1.png')
plt.close()

#*********************************PART (C) ***************************************

bestAlpha = alpha_ridge[errors_test.index(min(errors_test))]
print "The Best value received with this value pf Alpha: "
print bestAlpha


#********************************** PART (D) ***************************************

clf = RidgeCV(alphas=alpha_ridge,cv=5)
clf.fit(X_train, y_train)

print "Results for 5 fold " 
print clf.alpha_

#******************************* PART (E) ****************************************


means = numpy.linspace(-1,1,5)
variances = [0.1,0.5,1,5]

for m in means:
    test_err = OrderedDict({})
    train_err = OrderedDict({})
    for var in variances:
        transformed_X_train = numpy.zeros((len(X_train),len(X_train[0])))
        for i in range(0,len(X_train)):
            for j in range(0,len(X_train[i])):
                x = X_train[i][j]
                transformed_X_train[i][j] = (1/(sqrt(2*3.14*var)))* numpy.exp(((x-m)**2)/(2*var))
        lr = LinearRegression()
        lr.fit(transformed_X_train,y_train)
        y_train_pred = lr.predict(transformed_X_train)
        y_pred = lr.predict(X_test)
        train_err.update({var : sqrt(mean_squared_error(y_train, y_train_pred))})
        test_err.update({var : sqrt(mean_squared_error(y_test, y_pred))})
    plt.plot(train_err.keys(), train_err.values(), label='Train Error (mean: '+str(m)+')')
    plt.plot(test_err.keys(), test_err.values(), label='Test Error (mean: '+str(m)+')')

lr = LinearRegression()
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_pred = lr.predict(X_test)
lr_train_err = sqrt(mean_squared_error(y_train, y_train_pred))
lr_test_err = sqrt(mean_squared_error(y_test, y_pred))
train_line_data = numpy.array([lr_train_err for i in xrange(len(variances))])
test_line_data = numpy.array([lr_test_err for i in xrange(len(variances))])
plt.plot(variances, train_line_data, label='Train Error (linear regression)')
plt.plot(variances, test_line_data, label='Test Error(linear regression)')

# Put a legend to the right of the current axis
ax = plt.subplot(111)
plt.title('Root Mean Square Error as a function of the variance')
plt.xlabel('Variance')
plt.ylabel('RMSE')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('rms_vs_variance.png')
plt.close()


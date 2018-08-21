import numpy as np
import math
import sklearn

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
#np.set_printoptions(threshold=np.nan)
##################################################################


##################################################################
#Importing the data and putting it into an array
print "-----Dataset Manipuation-----"
print "Putting text file into array..."
alldata = []
with open('dataset.txt', 'r') as input_file:
    input_file.next()
    for line in input_file:
        row = line.split(',')
        alldata.append(row)
# print "alldata"
#instances 858
# print len(alldata)
#features 36
# print len(alldata[0])

for i in range(len(alldata)):
    alldata[i][35] = alldata[i][35].strip()
# print "alldata after"
#instances 858
# print len(alldata)
#features 36
# print len(alldata[0])
# print alldata

print "Converting alldata to numpy..."
alldata_numpy = np.asarray(alldata)
# print alldata_numpy

# print np.shape(alldata_numpy)

print "Replacing all missing values '?' with nan..."
#Replace missing values '?' with nan
for i in range(0, 858):
    for j in range(0, 36):
        if alldata_numpy[i][j] == '?':
            alldata_numpy[i][j] = float('nan')
#print alldata_numpy

print "Converting all numpy vals to floats..."
alldata_numpy = alldata_numpy.astype(float)
#print alldata_numpy

# OR use Imputer
# imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# alldata_numpynew = imp.fit_transform(alldata_numpy)
# print "imputer"
# print alldata_numpynew

#Replace nan with mean of column
print "Finding all indices where nan..."
# print "all indices where nan"
nan_indices = np.argwhere(np.isnan(alldata_numpy))
#print nan_indices
inds = np.where(np.isnan(alldata_numpy))
# print "inds"
# print inds
# print "all rows where there are nan values"
nan_indices_unique_rows = np.unique(nan_indices[:, 0])
# print nan_indices_unique_rows
# print np.shape(nan_indices_unique_rows)

# print "all cols where there are nan values"
nan_indices_unique_cols = np.unique(nan_indices[:, 1])
# print nan_indices_unique_cols

print "Finding the col mean..."
# print "col mean"
col_mean = np.nanmean(alldata_numpy, axis = 0)
# print np.shape(col_mean)
# print col_mean

# print "Finding the col median..."
# # print "col mean"
# col_median = np.median(alldata_numpy, axis = 0)
# print col_median

#Replaces all nan with mean of column
print "Replacing all nan with col mean..."
alldata_numpy[inds] = np.take(col_mean, inds[1])
#alldata_numpy[inds] = np.take(col_median, inds[1])
print "my method"
print alldata_numpy

print "Extracting target from dataset..."
y = alldata_numpy[:,35]

#Split up data into 80% train, 20% testing
print ""
print "-----Splitting up dataset-----"
print "Size of dataset is", np.shape(alldata_numpy)
print "Splitting dataset up into 80% train, 20% test..."
x_train, x_test, y_train, y_test = train_test_split(alldata_numpy, y, test_size = 0.2, random_state = 0)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

#The last four columns are the target variables
x_train = x_train[:,:32]
x_test = x_test[: ,:32]

##################################################################
#Preprocessing/Transformation
print "Doing data preprocessing and transformation..."
zscore_train_numpy = stats.zscore(x_train, axis = 1)
zscore_test_numpy = stats.zscore(x_test, axis = 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_numpy = scaler.fit_transform(x_train)
scaled_test_numpy = scaler.fit_transform(x_test)

scaler = StandardScaler()
standardized_train_numpy = scaler.fit_transform(x_train)
standardized_test_numpy = scaler.fit_transform(x_test)

scaler = Normalizer()
normalized_train_numpy = scaler.fit_transform(x_train)
normalized_test_numpy =  scaler.fit_transform(x_test)

scaler = Binarizer(threshold = 0.0)
binarized_train_numpy = scaler.fit_transform(x_train)
binarized_test_numpy = scaler.fit_transform(x_test)
##################################################################
#Feature Selection
print "Doing feature selection..."
sel = VarianceThreshold(threshold = ((.8 * (1 - .8))))
variance_train_numpy = sel.fit_transform(x_train)
variance_test_numpy = sel.fit_transform(x_test)

kbest_train_numpy = SelectKBest(chi2, k = 5).fit_transform(x_train, y_train)
kbest_test_numpy = SelectKBest(chi2, k = 5).fit_transform(x_test, y_test)

print np.shape(x_train), np.shape(x_test)
print np.shape(kbest_train_numpy), np.shape(kbest_test_numpy)

##################################################################
#Linear Discriminant Analysis
print ""
print "-----Linear Discriminant Analysis-----"
lda = LinearDiscriminantAnalysis()
lda.fit(kbest_train_numpy, y_train)
#lda.fit(x_train, y_train)
#myprediction1 =  lda.predict(x_test)
myprediction1 =  lda.predict(kbest_test_numpy)
# print myprediction1
# print np.shape(myprediction1), np.shape(y_test)
# print "ytest"
# print y_test
# print "myprediction"
# print myprediction1

#Use score to get the accuracy of the model
print "Linear Discriminant Analysis Accuracy..."
# score = lda.score(x_test, y_test)
score = accuracy_score(y_test, myprediction1)
print score
##################################################################
#Quadratic Discriminant Analysis
print ""
print "-----Quadratic Discriminant Analysis-----"
qda = QuadraticDiscriminantAnalysis()
qda.fit(kbest_train_numpy, y_train)
# qda.fit(x_train, y_train)
#myprediction2 = qda.predict(x_test)
myprediction2 = qda.predict(kbest_test_numpy)
# print myprediction2

#Use score to get the accuracy of the model
print "Quadratic Discriminant Analysis Accuracy..."
# score = qda.score(x_test, y_test)
score = accuracy_score(y_test, myprediction2)
print score

##################################################################
#Logistic Regression
print ""
print "-----Logistic Regression-----"
logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train, y_train)
logisticRegr.fit(kbest_train_numpy, y_train)
# myprediction3 = logisticRegr.predict(x_test)
myprediction3 = logisticRegr.predict(kbest_test_numpy)
# print np.shape(myprediction3), np.shape(y_test)
# print "ytest"
# print y_test
# print "myprediction"
# print myprediction3

#Use score to get the accuracy of the model
print "Logistic Regression Accuracy..."
# score = logisticRegr.score(x_test, y_test)
score = accuracy_score(y_test, myprediction3)
print score

##################################################################
#Gaussian Naive Bayes
#more sensitive to features? scale?
#try feature transformations
#look at the range between them (age)
#look at the scale of the columns
print ""
print "-----Gaussian Naive Bayes-----"
GaussNB = GaussianNB()
# GaussNB.fit(x_train, y_train)
GaussNB.fit(kbest_train_numpy, y_train)
# myprediction4 = GaussNB.predict(x_test)
myprediction4 = GaussNB.predict(kbest_test_numpy)
print np.shape(myprediction4), np.shape(y_test)
#print "ytest"
#print y_test
#print "myprediction"
#print myprediction4

#Use score to get the accuracy of the model
print "Gaussian Naive Bayes Accuracy..."
# score = GaussNB.score(x_test, y_test)
score = accuracy_score(y_test, myprediction4)
print score

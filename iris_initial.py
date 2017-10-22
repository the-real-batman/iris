#Loading libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Load Dataset using pandas
filename='iris.data'
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(filename,names=names)
#View dimensions of the data
print(dataset.shape)
#To print the descriptions
print(dataset.describe())
#Distribution by class
print(dataset.groupby('class').size())
#forming a histogram
dataset.hist()
plt.show()
#Split the dataset
array = dataset.values
X = array[:,0:4]  		#selects all the numeric values
Y = array[:,4]			#selects all the categories of flowers
validation_size = 0.20		#defines the proportion for validation dataset
seed = 7 
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)		#splits the dataset
#Test option and evaluation metric
seed=7
scoring='accuracy'
#Spot Check Algorithms
models= []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
#Evaluate each model in turn
results=[]
names=[]
for name,model in models:
	kfold=model_selection.KFold(n_splits=10,random_state=seed)	#Divide the dataset into folds
	cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg="%s:%f (%f)" % (name,cv_results.mean(),cv_results.std())
	print(msg)
#Comparing algorithms virtually
fig=plt.figure()
fig.suptitle('Algorithm Comparision')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#We now know that KNN was the best algo, now we use it
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv('wineQualityReds.csv')
data=data.iloc[:,1:]

data.to_csv('data.csv', index=False, header=False)

# x=data.iloc[:,0:-1]
# y=data.iloc[:,-1]



# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

# model = svm.LinearSVC(random_state=20)
# model = KNeighborsClassifier(n_neighbors=5)

# model.fit(x_train,y_train)
# predicted= model.predict(x_test)
# score=accuracy_score(y_test,predicted)
# print("Your Model Accuracy is", score)
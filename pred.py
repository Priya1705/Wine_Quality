import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.naive_bayes import GaussianNB 

data=pd.read_csv('data.csv')

# data.drop('total.sulfur.dioxide', axis=1, inplace=True)

fp_read = open('data.csv',"r")
fp_write = open('clean_data.csv',"w+")
fp_write2 = open('dirty_data.csv',"w+")
for line in fp_read:
	line = line.strip("\n")
	line = line.split(",")
	fixed_acidity=line[0]
	volatile_acidity=line[1]
	citric_acid=line[2]
	residual_sugar=line[3]
	chlorides=line[4]
	free_sulfur_dioxide=line[5]
	total_sulfur_dioxide=line[6]
	density=line[7]
	pH=line[8]
	sulphates=line[9]
	alcohol=line[10]
	quality=line[11]
	x=pd.Series(total_sulfur_dioxide)
	x=float(x)
	y=pd.Series(free_sulfur_dioxide)
	y=float(y)
	z=pd.Series(residual_sugar)
	z=float(z)
	m=pd.Series(fixed_acidity)
	m=float(m)
	a=120.0
	b=60.0
	c=3.5
	write_str = fixed_acidity +"," + volatile_acidity + "," + citric_acid + "," + residual_sugar + "," + chlorides+","+free_sulfur_dioxide+","+total_sulfur_dioxide+","+density+","+pH+","+sulphates+","+alcohol+","+quality+"\n"
	if(x <= a and y <= b and z <= c):
		fp_write.write(write_str)
	else:
		fp_write2.write(write_str)

	# if(z <= 2.0):
	# 	fp_write.write(write_str)
	# else:
	# 	fp_write2.write(write_str)
	# if(y <= 65.0):
	# 	fp_write.write(write_str)
	# else:
	# 	fp_write2.write(write_str)
	# if(x <= 120.0):
	# 	fp_write.write(write_str)
	# else:
	# 	fp_write2.write(write_str)
	# if(m <= 15.0):
	# 	fp_write.write(write_str)
	# else:
	# 	fp_write2.write(write_str)

dataset=pd.read_csv('clean_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=1/3,random_state=0)

model= tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
predicted=model.predict(x_test)

# support = svm.LinearSVC(random_state=20)
# support.fit(x_train, y_train)
# predicted= support.predict(x_test)

score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)


# regressor=LinearRegression()
# regressor.fit(x_train,y_train)

# y_pred=regressor.predict(x_test)
# test_rmse=(np.sqrt(mean_squared_error(y_test, y_pred)))
# test_r2=r2_score(y_test, y_pred)

# print(test_rmse)
# print(test_r2)








#plot graph to check difference between predicted and normal outcome

# fig,ax = plt.subplots()
# ax.scatter(y_test,y_pred)
# #ax.set_x_label('Measured')
# #ax.set_y_label('Predicted')
# ax.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()],'k--',lw=4)
# ax.set_title('Actual(x) vs Predicted(y)')
# fig.show()
# plt.savefig('Actual vs Predicted.pdf')
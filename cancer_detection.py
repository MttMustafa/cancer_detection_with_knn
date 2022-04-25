import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Python\AI\data.csv")

data.drop(["Unnamed: 32"], axis=1, inplace = True)
data.drop(["id"], axis=1 , inplace = True)

# print(data.isnull().sum())

# print(data.shape)

# print(data.diagnosis.value_counts())

# sns.countplot(data.diagnosis, label= "count")
# plt.show()


correlation = data.corr()

# print(correlation)


# plt.figure(figsize = (18,18))
# sns.heatmap(correlation,annot=True)
# plt.show()

benign = data[data.diagnosis == 'B']
malignant = data[data.diagnosis == 'M']


# plt.scatter(malignant.radius_mean,malignant.perimeter_mean, color = 'red')
# plt.scatter(benign.radius_mean,benign.perimeter_mean,color = 'blue')
# plt.xlabel("Radius Mean")
# plt.ylabel("Perimeter Mean")
# plt.show() 


# plt.scatter(malignant.radius_mean,malignant.fractal_dimension_mean, color = 'red')
# plt.scatter(benign.radius_mean,benign.fractal_dimension_mean,color = 'blue')
# plt.xlabel("Radius Mean")
# plt.ylabel("Perimeter Mean")
# plt.show() 

data.diagnosis = [1 if i == "B" else 0 for i in data.diagnosis]

y = data.diagnosis #classes

x = data.drop(["diagnosis"],axis = 1) #values without classes

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state = 44)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 7)
knn.fit(x_train,y_train)

print(knn.score(x_test,y_test))

prediction = knn.predict(x_test)
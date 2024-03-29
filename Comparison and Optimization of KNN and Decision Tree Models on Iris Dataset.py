'''
使用鳶尾花資料集進行以下處理
1.將全部資料使用KNN建立訓練模型(K設定為50)後，查看其模型正確率，並與決策樹(層數設定為4)的模型正確率做比較。
2.將資料分割成3:2的訓練資料集及測試資料集(亂數種子設定為50)，並使用KNN演算法。選擇適當的K個個數，並輸出其模型正確率。
'''
import pandas as pd
from sklearn import neighbors,datasets
from sklearn import tree
from sklearn.model_selection import train_test_split as tts

iris=datasets.load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
X.columns=['sepal_length','sepal_width','petal_length','petal_width']
target=pd.DataFrame(iris.target,columns=['target'])
y=target['target']

#Knn
knn=neighbors.KNeighborsClassifier(n_neighbors=50)
knn.fit(X,y)
print('KNN訓練模型正確率:',knn.score(X,y))

#決策樹
dtree=tree.DecisionTreeClassifier(max_depth=4)
dtree.fit(X,y)
print('決策樹訓練模型正確率:',dtree.score(X,y))

XTrain,XTest,yTrain,yTest=tts(X,y,test_size=0.4,random_state=50)

#k通常低於訓練樣本數的平方根，90**0.5=9.多，設定為整數10，(1,11)
a=[]
for k in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(XTrain, yTrain)
    accuracy = knn.score(XTest, yTest)
    a.append(accuracy)
a_max=a.index(max(a))+1
print(f'最適合的k值是{a_max},對應正確率為{max(a)}')
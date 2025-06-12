import sklearn
import pandas as pd
import matplotlib.pyplot as mp
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import seaborn as sn
import streamlit as st





df = pd.read_csv(r"C:\Users\anish\Downloads\ml_practice_data_with_nans.csv")

for ele in df:
    df[ele] = df[ele].fillna(math.floor(df[ele].median()))

df.iloc[0,5] = 2

df.iloc[range(0, 101), 5] = 0
df.iloc[range(100, 200), 5] =1
print(df.tail(15))


model = sklearn.tree.DecisionTreeClassifier(criterion='entropy')

x, xx, y, yy = sklearn.model_selection.train_test_split(df[['feature_1','feature_2', 'feature_3','feature_4','feature_5']], df.target, test_size=.5)

model.fit(x, y)

print(model.score(x, y))
print(model.predict_proba([[-0.848758,-1.000000,-1.523187,1.195047,-1.460698]]))

confus = sklearn.metrics.confusion_matrix(yy, model.predict(xx))
fig,ax = plt.subplots()
sn.heatmap(confus, annot=True)
st.pyplot(fig)















# df = pd.read_csv(r"C:\Users\anish\Downloads\titanic.csv")


# for ele in ['PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked']:
#     df = df.drop(ele, axis=1)

# le = sklearn.preprocessing.LabelEncoder()

# df['Sex'] = le.fit_transform(df['Sex'])

# model = sklearn.tree.DecisionTreeClassifier(criterion='gini', random_state=9)

# X = df[['Pclass', 'Sex', 'Age','Fare']]
# Y = df.Survived
# x, xx, y, yy = sklearn.model_selection.train_test_split(X, Y, test_size=.5)

# model.fit(x,y)
# print(df.head())


# print(model.predict_proba([[1, 1, 20, 40]]))
# print(model.score(xx, yy))
# consu = sklearn.metrics.confusion_matrix(yy, model.predict(xx))

# sn.heatmap(consu, annot=True)
# mp.show()

























# Y = df.salary_more_then_100k
# df = df.drop('salary_more_then_100k', axis=1)

# le = sklearn.preprocessing.LabelEncoder()

# df['company'] = le.fit_transform(df['company'])
# df['job'] = le.fit_transform(df['job'])
# df['degree'] = le.fit_transform(df['degree'])

# model = sklearn.tree.DecisionTreeClassifier()

# print(model)

# X = df[['company','job','degree']]

# x, xx, y, yy = sklearn.model_selection.train_test_split(X, Y, test_size=.5)

# model.fit(x, y)
# print(x)
# print(xx)


# confuns = sklearn.metrics.confusion_matrix(yy, model.predict(xx))


# print(confuns)


# sn.heatmap(confuns, annot=True)
# mp.show()


































































#
















































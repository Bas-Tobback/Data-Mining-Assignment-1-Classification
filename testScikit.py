# Test file to explore scikit learn and pandas, this has no value to the project,
# it is just to save some parts or test the out

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# sending cost
send_cost = 10

# answer rates of income categories
low_income_answer_rate = 5/100
high_income_answer_rate = 10/100

# average profit
low_profit = -310
high_profit = 980

# read the Excel files with the database
existing_customers = pd.read_excel(r'./data/existing-customers.xlsx')
potential_customers = pd.read_excel(r'./data/potential-customers.xlsx')

# take the features
features = ['RowID', 'age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
X = existing_customers[features]
X_predict = potential_customers[features]

# take the labels
y = existing_customers['class']

# transform columns to ensure the fitting can go smoothly, no strings should be passed
# enc = OrdinalEncoder()
# enc.fit(X)
# X = enc.transform(X)

preprocess = make_column_transformer(
    (OrdinalEncoder(), ['RowID', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']),
    remainder='passthrough'
)

X = preprocess.fit_transform(X)

# create a pipeline object
im = IterativeImputer()
X = im.fit_transform(X)

pipe = RandomForestClassifier()
gauss = GaussianNB()
cate = CategoricalNB()
deci = DecisionTreeClassifier()


# load the iris dataset and split it into train and test sets
# X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# fit the whole pipeline
pipe.fit(X_train, y_train)
gauss.fit(X_train, y_train)
cate.fit(X_train, y_train)
deci.fit(X_train, y_train)

# we can now use it like any other estimator
# print("RandomForest report :" + str(classification_report(y_test, pipe.predict(X_test))))
#
# print("GaussianBayes accuracy :" + str(classification_report(y_test, gauss.predict(X_test))))
# # Categorical gives errors when not using the right random state
# # print("CategoricalBayes accuracy :" + str(accuracy_score(y_test, cate.predict(X_test))))
# print("DecisionTree accuracy :" + str(classification_report(y_test, deci.predict(X_test))))


# print(existing_customers.head())
# print(potential_customers.head())

# enc2 = OrdinalEncoder()
# enc2.fit(X_predict)
X_predict = preprocess.transform(X_predict)
X_predict = im.transform(X_predict)

predictions = pipe.predict_proba(X_predict)

total_profit = 0
sending_customers = []

print(len(predictions))
print(len(X_predict))

for i in range(0, len(predictions)):
    low = predictions[i][0]
    high = predictions[i][1]
    value = low * low_income_answer_rate * low_profit + high * high_income_answer_rate * high_profit - send_cost
    if value >= 0:
        total_profit += value
        sending_customers.append(potential_customers['RowID'].values[i])

print(total_profit)
print(len(sending_customers))

f = open("customer_ids.txt", "w")
for cid in sending_customers:
    f.write(cid)
    f.write("\n")
f.close()


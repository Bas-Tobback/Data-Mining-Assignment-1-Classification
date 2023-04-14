# Test file to explore scikit learn and pandas, this has no value to the project,
# it is just to save some parts or test the out

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# read the excel files with the database
existing_customers = pd.read_excel(r'./data/existing-customers.xlsx')
df2 = pd.read_excel(r'./data/potential-customers.xlsx')

# take the features
features = ['RowID', 'age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
X = existing_customers[features]

# take the labels
y = existing_customers['class']

# transform columns to ensure the fitting can go smoothly, no strings should be passed
enc = OrdinalEncoder()
enc.fit(X)
X = enc.transform(X)

# create a pipeline object
pipe = make_pipeline(
    IterativeImputer(),
    CategoricalNB()
)

# load the iris dataset and split it into train and test sets
# X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# fit the whole pipeline
pipe.fit(X_train, y_train)

# we can now use it like any other estimator
print(accuracy_score(y_test, pipe.predict(X_test)))

print(existing_customers.head())

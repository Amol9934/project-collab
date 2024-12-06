# project-collab
machine learning project

Open In Colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')
     
Data collection and processing


# loading the dataset to panda DataFrame
loan_dataset = pd.read_csv('/content/data.csv')
     

type(loan_dataset)
     
pandas.core.frame.DataFrame
def __init__(data=None, index: Axes | None=None, columns: Axes | None=None, dtype: Dtype | None=None, copy: bool | None=None) -> None
Two-dimensional, size-mutable, potentially heterogeneous tabular data.

Data structure also contains labeled axes (rows and columns).
Arithmetic operations align on both row and column labels. Can be
thought of as a dict-like container for Series objects. The primary
pandas data structure.

Parameters
----------
data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
    Dict can contain Series, arrays, constants, dataclass or list-like objects. If
    data is a dict, column order follows insertion-order. If a dict contains Series
    which have an index defined, it is aligned by its index. This alignment also
    occurs if data is a Series or a DataFrame itself. Alignment is done on
    Series/DataFrame inputs.

    If data is a list of dicts, column order follows insertion-order.

index : Index or array-like
    Index to use for resulting frame. Will default to RangeIndex if
    no indexing information part of input data and no index provided.
columns : Index or array-like
    Column labels to use for resulting frame when data does not have them,
    defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
    will perform column selection instead.
dtype : dtype, default None
    Data type to force. Only a single dtype is allowed. If None, infer.
copy : bool or None, default None
    Copy data from inputs.
    For dict data, the default of None behaves like ``copy=True``.  For DataFrame
    or 2d ndarray input, the default of None behaves like ``copy=False``.
    If data is a dict containing one or more Series (possibly of different dtypes),
    ``copy=False`` will ensure that these inputs are not copied.

    .. versionchanged:: 1.3.0

See Also
--------
DataFrame.from_records : Constructor from tuples, also record arrays.
DataFrame.from_dict : From dicts of Series, arrays, or dicts.
read_csv : Read a comma-separated values (csv) file into DataFrame.
read_table : Read general delimited file into DataFrame.
read_clipboard : Read text from clipboard into DataFrame.

Notes
-----
Please reference the :ref:`User Guide <basics.dataframe>` for more information.

Examples
--------
Constructing DataFrame from a dictionary.

>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df
   col1  col2
0     1     3
1     2     4

Notice that the inferred dtype is int64.

>>> df.dtypes
col1    int64
col2    int64
dtype: object

To enforce a single dtype:

>>> df = pd.DataFrame(data=d, dtype=np.int8)
>>> df.dtypes
col1    int8
col2    int8
dtype: object

Constructing DataFrame from a dictionary including Series:

>>> d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
>>> pd.DataFrame(data=d, index=[0, 1, 2, 3])
   col1  col2
0     0   NaN
1     1   NaN
2     2   2.0
3     3   3.0

Constructing DataFrame from numpy ndarray:

>>> df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
...                    columns=['a', 'b', 'c'])
>>> df2
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9

Constructing DataFrame from a numpy ndarray that has labeled columns:

>>> data = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
...                 dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")])
>>> df3 = pd.DataFrame(data, columns=['c', 'a'])
...
>>> df3
   c  a
0  3  1
1  6  4
2  9  7

Constructing DataFrame from dataclass:

>>> from dataclasses import make_dataclass
>>> Point = make_dataclass("Point", [("x", int), ("y", int)])
>>> pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
   x  y
0  0  0
1  0  3
2  2  3

Constructing DataFrame from Series/DataFrame:

>>> ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
>>> df = pd.DataFrame(data=ser, index=["a", "c"])
>>> df
   0
a  1
c  3

>>> df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"], columns=["x"])
>>> df2 = pd.DataFrame(data=df1, index=["a", "c"])
>>> df2
   x
a  1
c  3

# printing the first 5 rows of the dataframe
loan_dataset.head()
     
Loan_ID	Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area	Loan_Status
0	LP001003	Male	Yes	1	Graduate	No	4583	1508.0	128.0	360.0	1.0	Rural	N
1	LP001005	Male	Yes	0	Graduate	Yes	3000	0.0	66.0	360.0	1.0	Urban	Y
2	LP001006	Male	Yes	0	Not Graduate	No	2583	2358.0	120.0	360.0	1.0	Urban	Y
3	LP001008	Male	No	0	Graduate	No	6000	0.0	141.0	360.0	1.0	Urban	Y
4	LP001013	Male	Yes	0	Not Graduate	No	2333	1516.0	95.0	360.0	1.0	Urban	Y

# number of rows and columns
loan_dataset.shape
     
(381, 13)

# statistical measures
loan_dataset.describe()

     
ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History
count	381.000000	381.000000	381.000000	370.000000	351.000000
mean	3579.845144	1277.275381	104.986877	340.864865	0.837607
std	1419.813818	2340.818114	28.358464	68.549257	0.369338
min	150.000000	0.000000	9.000000	12.000000	0.000000
25%	2600.000000	0.000000	90.000000	360.000000	1.000000
50%	3333.000000	983.000000	110.000000	360.000000	1.000000
75%	4288.000000	2016.000000	127.000000	360.000000	1.000000
max	9703.000000	33837.000000	150.000000	480.000000	1.000000

# number of missing values in each column
loan_dataset.isnull().sum()
     
0
Loan_ID	0
Gender	5
Married	0
Dependents	8
Education	0
Self_Employed	21
ApplicantIncome	0
CoapplicantIncome	0
LoanAmount	0
Loan_Amount_Term	11
Credit_History	30
Property_Area	0
Loan_Status	0

dtype: int64

# dropping the missing values
loan_dataset = loan_dataset.dropna()
     

# number of missing values in each column
loan_dataset.isnull().sum()
     
0
Loan_ID	0
Gender	0
Married	0
Dependents	0
Education	0
Self_Employed	0
ApplicantIncome	0
CoapplicantIncome	0
LoanAmount	0
Loan_Amount_Term	0
Credit_History	0
Property_Area	0
Loan_Status	0

dtype: int64

#level encoding
loan_dataset.replace({"Loan_status":{'N':0,'Y':1}},inplace=True)
     

# printing the first 5 rows of the dataframe
loan_dataset.head()
     
Loan_ID	Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area	Loan_Status
0	LP001003	Male	Yes	1	Graduate	No	4583	1508.0	128.0	360.0	1.0	Rural	N
1	LP001005	Male	Yes	0	Graduate	Yes	3000	0.0	66.0	360.0	1.0	Urban	Y
2	LP001006	Male	Yes	0	Not Graduate	No	2583	2358.0	120.0	360.0	1.0	Urban	Y
3	LP001008	Male	No	0	Graduate	No	6000	0.0	141.0	360.0	1.0	Urban	Y
4	LP001013	Male	Yes	0	Not Graduate	No	2333	1516.0	95.0	360.0	1.0	Urban	Y

# Dependent column values
loan_dataset["Dependents"].value_counts()
     
count
Dependents	
0	194
2	47
1	43
3+	24

dtype: int64

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4 )
     

# dependent values
loan_dataset["Dependents"].value_counts()
     
count
Dependents	
0	194
2	47
1	43
4	24

dtype: int64
Data visualization

# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)
     
<Axes: xlabel='Education', ylabel='count'>


# marital status & loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
     
<Axes: xlabel='Married', ylabel='count'>


# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'property_area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
     

loan_dataset.head()

     
Loan_ID	Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area	Loan_Status
0	LP001003	1	1	1	1	0	4583	1508.0	128.0	360.0	1.0	Rural	N
1	LP001005	1	1	0	1	1	3000	0.0	66.0	360.0	1.0	Urban	Y
2	LP001006	1	1	0	0	0	2583	2358.0	120.0	360.0	1.0	Urban	Y
3	LP001008	1	0	0	1	0	6000	0.0	141.0	360.0	1.0	Urban	Y
4	LP001013	1	1	0	0	0	2333	1516.0	95.0	360.0	1.0	Urban	Y

# separating the data and lebel
X=loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_dataset['Loan_Status']
     

print(X)
print(Y)
     
     Gender  Married Dependents  Education  Self_Employed  ApplicantIncome  \
0         1        1          1          1              0             4583   
1         1        1          0          1              1             3000   
2         1        1          0          0              0             2583   
3         1        0          0          1              0             6000   
4         1        1          0          0              0             2333   
..      ...      ...        ...        ...            ...              ...   
376       1        1          4          1              0             5703   
377       1        1          0          1              0             3232   
378       0        0          0          1              0             2900   
379       1        1          4          1              0             4106   
380       0        0          0          1              1             4583   

     CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History  \
0               1508.0       128.0             360.0             1.0   
1                  0.0        66.0             360.0             1.0   
2               2358.0       120.0             360.0             1.0   
3                  0.0       141.0             360.0             1.0   
4               1516.0        95.0             360.0             1.0   
..                 ...         ...               ...             ...   
376                0.0       128.0             360.0             1.0   
377             1950.0       108.0             360.0             1.0   
378                0.0        71.0             360.0             1.0   
379                0.0        40.0             180.0             1.0   
380                0.0       133.0             360.0             0.0   

    Property_Area  
0           Rural  
1           Urban  
2           Urban  
3           Urban  
4           Urban  
..            ...  
376         Urban  
377         Rural  
378         Rural  
379         Rural  
380     Semiurban  

[308 rows x 11 columns]
0      N
1      Y
2      Y
3      Y
4      Y
      ..
376    Y
377    Y
378    Y
379    Y
380    N
Name: Loan_Status, Length: 308, dtype: object
Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)
     
(308, 11) (277, 11) (31, 11)
Traning the model:

Support vector Machine Model

# Assuming X_train and Y_train have categorical values
# Convert categorical variables in X_train to numerical using one-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)

# If Y_train is categorical, convert it to numerical using factorization
if Y_train.dtype == 'object':
    Y_train = pd.factorize(Y_train)[0]

# Fit the SVM model

classifier = svm.SVC(kernel='linear')
     

# trainning the support Vector machine model
classifier.fit(X_train,Y_train)
     
SVC(kernel='linear')
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Model Evaluation

# accuracy score on trainning data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

print("Accuracy on training data: ",training_data_accuracy)

     
Accuracy on training data:  0.8303249097472925

# accuracy score on trainning score
classifier.fit(X_train, Y_train)

# Make predictions on test data
X_test_prediction = classifier.predict(X_test)

# Check shapes of Y_test and predictions
print(X_test_prediction.shape, Y_test.shape)

# Calculate accuracy score
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on test data:", test_data_accuracy)
     
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-1-21a22f963f64> in <cell line: 2>()
      1 # accuracy score on trainning score
----> 2 classifier.fit(X_train, Y_train)
      3 
      4 # Make predictions on test data
      5 X_test_prediction = classifier.predict(X_test)

NameError: name 'classifier' is not defined

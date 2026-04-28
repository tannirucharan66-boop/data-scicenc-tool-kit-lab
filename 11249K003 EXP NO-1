EXPNO1 
1.1(A) Creating Arrays 
import numpy as np print(
"1D Array:\n", np.array([1,2,3,4,5]), "\n\n"
"2DArray:\n", np.array([[1,2,3],[4,5,6]]), "\n\n" "Zeros:\n", np.zeros((3,3)),
"\n\n"
"Ones:\n", np.ones((2,4)), "\n\n"
"Identity:\n", np.eye(3), "\n\n"
"Range:\n", np.arange(0,10,2)
)

1.1 (B) Array Operations
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6]) print(
"Array a:", a, "\n"
"Array b:", b, "\n\n"
"Addition (a + b):", a + b, "\n" "Multiplication (a * b):", a * b, "\n"
"Scalar Multiplication (a * 10):", a * 10, "\n\n" "Square Root of a:", np.sqrt(a),
"\n"
"Mean of a:", np.mean(a), "\n"
"Dot Product (a · b):", np.dot(a, b)
)

1.1(C) Indexing and Slicing
import numpy as np

matrix = np.array([[10, 20, 30],
[40, 50, 60],
[70, 80, 90]])

print(
"Matrix:\n", matrix, "\n\n"
"Element at [1,2]:", matrix[1, 2], "\n\n"
"First Row:", matrix[0, :], "\n"
"Second Column:", matrix[:, 1], "\n\n"
"Top-left 2x2 Sub-matrix:\n", matrix[0:2, 0:2]
)

1.2(A) Creating DataFrames
import pandas as pd
# From a Dictionary data = {
'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35],
'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
# Viewing data
print(df.head()) # First 5 rows print("\n

")

print(df.info()) # Summary of data types and non-nulls print("\n

")
print(df.describe()) # Statistical summary (mean,std, min, max)

1.2(B) Selection and Filtering
# Selecting a column ages =
df['Age'] print("Agescolumn:")
print(ages)
print("\n ")
# Filtering rows
above_25 = df[df['Age'] > 25] print("People with Age >
25:") print(above_25)
print("\n ")

# iloc (integer-based) row_0 =
df.iloc[0]

print("First row using iloc (row index 0):") print(row_0)
print("\n ")
# loc (label-based)
specific_val = df.loc[0, 'Name'] print("City of the person at
index 0:") print(specific_val)

1.2(C) Data Cleaning (Handling Missing Values)
import pandas as pd import numpy
as np
# ⃣Create sample dataset with missing values data = {
"Name": ["Alice", "Bob", "Charlie", "David", "Eva"], "Age": [23, np.nan, 22, 28,
np.nan],
"City": ["New York", "London", np.nan, "Paris", "Berlin"]
}
df = pd.DataFrame(data)
print("OriginalDataFrame:") print(df)
print("\n \n")
# ⃣Check for null values
print("Null values count in each column:") print(df.isnull().sum())
print("\n \n")
# ⃣Drop rows with missing values df_clean =
df.dropna()
print("DataFrame after dropping rows with missing values:") print(df_clean)
print("\n \n")
# ⃣Fill missing values with 0 df_filled = df.fillna(0)
print("DataFrame after filling missing values with 0:") print(df_filled)
print("\n \n")
# ⃣Fill missing 'Age' values with mean of the column df_mean_filled = df.copy()
df_mean_filled["Age"] = df_mean_filled["Age"].fillna(df_mean_filled["Age"].mean()) print("DataFrame
after filling 'Age' with mean value:") print(df_mean_filled)
print("\n \n")

# ⃣Fill missing 'City' values with a placeholder df_city_filled = df.copy()
df_city_filled["City"]=df_city_filled["City"].fillna("Unknown") print("DataFrame after filling 'City' with
'Unknown':") print(df_city_filled)

1.3 Matplotlib (Plotting Library)
import pandas as pd
import matplotlib.pyplot as plt import numpy as np
#Create dataset data = {
'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8],
'ExamScore': [35, 40, 50, 55, 65, 70, 78, 85]
}
df = pd.DataFrame(data) #
# ⃣Scatter Plot
#
plt.scatter(df['StudyHours'],df['ExamScore']) plt.xlabel("Study Hrs")
plt.ylabel("ExamScore")
plt.title("Study Hours vs Exam Score (Scatter Plot)") plt.show()
#
# ⃣Line Plot #
plt.plot(df['StudyHours'],df['ExamScore'],marker='o') plt.xlabel("Study Hours")
plt.ylabel("ExamScore")
plt.title("Exam Score Progress(Line Plot)") plt.show()
#

# ⃣Histogram #
plt.hist(df['ExamScore'],bins=2) plt.xlabel("Score Range")
plt.ylabel("Number of Students")
plt.title("ExamScore Distribution (Histogram)") plt.show()
#
# ⃣Bar Chart #
plt.bar(df['StudyHours'],df['ExamScore']) plt.xlabel("Study
Hours") plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score (Bar Chart)") plt.show()

1.4
# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
fromsklearn.model_selection import train_test_split from sklearn.linear_model
import LinearRegression from sklearn.metrics import mean_squared_error
# STEP 2: CREATE DATASET
data = {
'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'], 'Classes_Attended': [30, 35, 40, 45, 50, 55, 60,
65, 70, 75],
'Internal_Marks': [35, 38, 42, 46, 50, 55, 60, 65, 70, 75]
}
df = pd.DataFrame(data) print(df)
# STEP 3: FEATURES & TARGET
X = df[['Classes_Attended']] y =
df['Internal_Marks']
# STEP 4: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,
random_state=7
)
# STEP 5: TRAIN MODEL
model = LinearRegression()
model.fit(X_train,y_train) # STEP 6:
PREDICTION
predictions=model.predict(X_test) # STEP 7:
EVALUATION
mse =mean_squared_error(y_test, predictions) print("MSE:", mse)
print("Marks per Class:", model.coef_[0]) print("Base Marks:",
model.intercept_)
# STEP 8: VISUALIZATION
plt.scatter(X, y)
plt.plot(X, model.predict(X)) plt.xlabel("Classes Attended")
plt.ylabel("Internal Marks") plt.title("Classes Attended vsInternal
Marks") plt.show( )

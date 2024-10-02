# Upgrade
Recommender system using ML-KNN algorithm to improve students learning experiences 

The critical challenge is to identify the effective methods to enhance student learning outcomes. Some students learn from their mistakes, but the question is how they can be guided to do so efficiently. Without a proper guidance, students may choose less effective ways to correct their errors. Thus losing their chance to gain more marks. Hence there is a need to have a recommender system that can improve students learning experience.

![Screenshot 2024-10-02 112743](https://github.com/user-attachments/assets/d4faa51b-24e2-49ad-a502-9a1a20a48e31)

#!/usr/bin/env python
# coding: utf-8

# # Build Dataset

# In[1]:


# Import Libraries 

import pandas as pd
import random


# In[2]:


# Load the dataset file path

Train_Dataset_Path = 'Train_StudentPerformance.xlsx'
course_id = "19CSE301"


# In[3]:


# Generate synthetic data for a specified number of rows

df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
Assessments = list(df_courses['Assessments'])
Converted_Marks = list(df_courses['Converted Marks'].values)
        
max_scores = dict(zip(Assessments, Converted_Marks))
total_max_score = sum(max_scores.values())
threshold_score = 0.75 * total_max_score

Strategies = list(df_courses['Strategies'])
Assessments_strategy = dict(zip(Assessments, Strategies))

df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)        
Assessments = list(df_courses['Assessments'])
Converted_Marks = list(df_courses['Converted Marks'].values)
max_scores = dict(zip(Assessments, Converted_Marks))
total_max_score = sum(max_scores.values())
threshold_score = 0.75 * total_max_score
        
def generate_synthetic_data(row_count):
            data = []
            for i in range(row_count):
                row = {
                    'Student Id': 22000 + i,  # Generate student IDs incrementally
                    'Class': 'CSE A',  # Assuming class is the same for all
                }
                for assessment in max_scores.keys():
                    row[assessment] = random.randint(0, max_scores[assessment])
                data.append(row)
            return pd.DataFrame(data)

row_count = 100000
df = generate_synthetic_data(row_count)
df.to_excel(Train_Dataset_Path, index=False)


# In[4]:


# Generate the recommendations

def generate_recommendations_based_on_total(row):

    total_score = 0
    for _ , mark in row.items():
        if _ not in ['Student Id', 'Class']:
            total_score += mark

    if total_score < threshold_score:
        recommendations = []
        for assessment, con_marks in max_scores.items():
            if row[assessment] < 0.75 * con_marks:
                recommendations.append(Assessments_strategy[assessment])
        
        return "; ".join(recommendations) if recommendations else "Improve performance"
    
    elif (total_score >= threshold_score) and (total_score < 0.85 * total_max_score):
        recommendations = ["Good performance overall"]
        for assessment, con_marks in max_scores.items():
            if row[assessment] < 0.75 * con_marks:
                recommendations.append(Assessments_strategy[assessment])
        
        return "; ".join(recommendations) if recommendations else "Improve performance"
    
    elif (total_score >= 0.85 * total_max_score) and (total_score < 0.9 * total_max_score):
        recommendations = ["Excelent performance overall"]
        for assessment, con_marks in max_scores.items():
            if row[assessment] < 0.75 * con_marks:
                recommendations.append(Assessments_strategy[assessment])
        
        return "; ".join(recommendations) if recommendations else "Improve performance"
    
    elif (total_score >= 0.9 * total_max_score):
        return "Outstanding performance overall"

df['Recommendation'] = df.apply(generate_recommendations_based_on_total, axis=1)
df.to_excel(Train_Dataset_Path, index=False)
print(f"Dataset created successfully.")


# # Train

# #### Enable Multi Label to pass as a single parameter

# In[5]:


import skmultilearn
file = skmultilearn.__file__
file_path = file[:-11] + "adapt\\mlknn.py"

with open(file_path, 'r') as file:
    file_contents = file.read()

modified_contents = file_contents.replace('NearestNeighbors(self.k)', 'NearestNeighbors(n_neighbors=self.k)')

with open(file_path, 'w') as file:
    file.write(modified_contents)

print("File modified and saved successfully.")


# #### Model Training Phase

# In[6]:


import pandas as pd
import numpy as np
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_excel(Train_Dataset_Path)

# Define features and labels
X = df[Assessments]

# Preprocess the recommendations to convert them into multiple binary labels
# Here we assume that each recommendation has a unique string and split them
df['Recommendation'] = df['Recommendation'].fillna("")
df['Recommendation'] = df['Recommendation'].apply(lambda x: x.split("; ") if x else [])

# MultiLabelBinarizer to convert string labels into binary form
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['Recommendation'])

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

best_k = None
best_accuracy = 0.0

# Loop through a range of k values to find the optimal one
for k in range(22, 26):  # Testing k from 1 to 20
    mlknn = MLkNN(k=k)
    mlknn.fit(X_train, Y_train)
    
    # Predict on the test data
    Y_pred = mlknn.predict(X_test)
    
    # Evaluate the model's accuracy
    accuracy = accuracy_score(Y_test, Y_pred.toarray())
    print(f"k={k}, Accuracy: {accuracy * 100:.2f}%")
    
    # Update the best k and accuracy if this one is better
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

# Print the final result
print(f"\nBest k value: {best_k} with Accuracy: {best_accuracy * 100:.2f}%")


# #### Download the Trained model

# In[7]:


import joblib

model_file_path = 'train_model.joblib'
joblib.dump(mlknn, model_file_path)
mlb_file_path = 'train_mlb.joblib'
joblib.dump(mlb, mlb_file_path)  # Save the MultiLabelBinarizer

print(f"Model saved as {model_file_path}")
print(f"MLB saved as {mlb_file_path}")



# # Code Initialising

# In[8]:


# Libraries

import numpy as np
import pandas as pd


# In[9]:


# Course Code Entry

#course_id = input("Enter Course ID: ")
course_id = "19CSE301"


# In[10]:


# Load the Student Dataset

df_test = pd.read_excel('Students.xlsx',sheet_name=course_id)

# Load the Course Dataset

df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)


# In[11]:


# View Head of Student dataset 

df_test.head()


# In[12]:


# View Course dataset 

df_courses


# # Data Preprocessing

# In[13]:


df_test.isnull().sum()


# In[14]:


Assessments = df_courses['Assessments']
print(Assessments)


# In[15]:


for i in Assessments:
        df_test[i].fillna(0, inplace=True)
        df_test[i] = df_test[i].astype(int)
df_test


# In[16]:


df_test.to_excel('Students.xlsx', sheet_name=course_id, index=False)


# #### Convert Marks

# In[17]:


Total_Marks = list(df_courses['Total Marks'].values)
Converted_Marks = list(df_courses['Converted Marks'].values)
print(Total_Marks)
print(Converted_Marks)


# In[18]:


Converted_Assessments_name = []

for i,j in enumerate(Assessments):
    converted_column_name = j + ' Converted'
    Converted_Assessments_name.append(converted_column_name)
    df_test[converted_column_name] = round((df_test[j] * Converted_Marks[i])/ Total_Marks[i])
    df_test[converted_column_name] = df_test[converted_column_name].astype(int)
    
df_test.to_excel('Students.xlsx', sheet_name=course_id, index=False)
df_test


# In[19]:


for i,j in enumerate(Assessments):
    df_test['Total'] = df_test[Converted_Assessments_name].sum(axis=1)
    
df_test.to_excel('Students.xlsx', sheet_name=course_id, index=False)
df_test


# # Test

# In[20]:


import pandas as pd
import joblib  # or import pickle

# Load the trained MLkNN model
model_file_path = 'train_model.joblib'  # or use 'mlknn_model.pkl' if you used pickle
loaded_model = joblib.load(model_file_path)

df_students = pd.read_excel('Students.xlsx',sheet_name=course_id)

# Prepare your test input data
# Make sure the test data is structured like the training data
recommendations_list = []

for index, row in df_students.iterrows():
    test_data = {assessment[:-10]: [f"{row[assessment]:.2f}"] for assessment in Converted_Assessments_name}
    test_df = pd.DataFrame(test_data)

    mlb = joblib.load(mlb_file_path) 
    predictions = loaded_model.predict(test_df)
    predicted_labels = mlb.inverse_transform(predictions.toarray())

    recommendation = ""
    for i, labels in enumerate(predicted_labels):
        if labels:
            recommendation += f"{', '.join(labels)}; "
        else:
            recommendation += "No Recommendations; "
    recommendations_list.append(recommendation)

df_students['Recommendations'] = recommendations_list

df_students.to_excel('Students.xlsx', sheet_name=course_id, index=False)

print("Recommendations done")


# In[21]:


df_test = pd.read_excel('Students.xlsx',sheet_name=course_id)
df_test.head()


# In[ ]:




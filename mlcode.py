import pandas as pd
import numpy as np
import random
from tkinter import messagebox
from openpyxl import load_workbook
import os

from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib


#==========================================================================================================================
#Excel add

course_excel_filename = "Courses.xlsx"
Test_excel_filename = "Students.xlsx"


#==========================================================================================================================
#Generate Dataset

def generate_data(course_id):
    
    Train_Dataset_Path = course_id + "_train_dataset.xlsx"
    
    if os.path.isfile(Train_Dataset_Path):
        print(f"{Train_Dataset_Path} exists.")
    else:
        print(f"{Train_Dataset_Path} does not exist. So creating the Training dataset")
        
        df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
        
        Assessments = list(df_courses['Assessments'])
        Converted_Marks = list(df_courses['Converted Marks'].values)
        
        max_scores = dict(zip(Assessments, Converted_Marks))
        total_max_score = sum(max_scores.values())
        threshold_score = 0.75 * total_max_score
        
        Strategies = list(df_courses['Strategies'])
        Assessments_strategy = dict(zip(Assessments, Strategies))
        
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


#==========================================================================================================================
#Traing Phase

def training_data(course_id):
    model_file_path = course_id + "_dataset_model.joblib"
    mlb_file_path = course_id + '_dataset_mlb.joblib'

    if os.path.isfile(model_file_path):
        print(f"{model_file_path} exists.")
    else:
        print(f"{model_file_path} does not exist. So creating the model")
        
        Train_Dataset_Path = course_id + "_train_dataset.xlsx"
        
        df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
        Assessments = list(df_courses['Assessments'])
    
        df = pd.read_excel(Train_Dataset_Path)

        X = df[Assessments]

        df['Recommendation'] = df['Recommendation'].fillna("")
        df['Recommendation'] = df['Recommendation'].apply(lambda x: x.split("; ") if x else [])

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(df['Recommendation'])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        mlknn = MLkNN(k=30)

        mlknn.fit(X_train, Y_train)

        Y_pred = mlknn.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred.toarray())
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        
        joblib.dump(mlknn, model_file_path)
        print(f"Model saved as {model_file_path}")
        joblib.dump(mlb, mlb_file_path)
        print(f"MLB saved as {mlb_file_path}")


#==========================================================================================================================
#Data Preprocessing

def dataprep(course_id):
    global Converted_Assessments_name
    df_courses = pd.read_excel(course_excel_filename, sheet_name=course_id)
    df_test = pd.read_excel(Test_excel_filename, sheet_name=course_id)

    Assessments = df_courses['Assessments']
    for i in Assessments:
        df_test[i].fillna(int(0), inplace=True)
        #df_test[i] = df_test[i].astype(int)
        df_test[i] = df_test[i]
    
    df_test.to_excel(Test_excel_filename, sheet_name=course_id, index=False)
    
    Total_Marks = list(df_courses['Total Marks'].values)
    Converted_Marks = list(df_courses['Converted Marks'].values)
    
    Converted_Assessments_name = []
    
    for i,j in enumerate(Assessments):
        converted_column_name = j + ' Converted'
        Converted_Assessments_name.append(converted_column_name)
        df_test[converted_column_name] = round((df_test[j] * Converted_Marks[i])/ Total_Marks[i])
        df_test[converted_column_name] = df_test[converted_column_name].astype(int)

    for i,j in enumerate(Assessments):
        df_test['Total'] = df_test[Converted_Assessments_name].sum(axis=1)
        
    df_test.to_excel(Test_excel_filename, sheet_name=course_id, index=False)


#==========================================================================================================================
#Testing Phase
#==========================================================================================================================
#========================================================================================================================== 
#Student Analysis

def student_recommendation(l):
    course_id = l[0]
    
    generate_data(course_id)
    training_data(course_id)
    dataprep(course_id)
    
    df_students = pd.read_excel(Test_excel_filename,sheet_name=course_id)
    df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
    
    model_file_path = course_id + "_dataset_model.joblib"
    mlb_file_path = course_id + '_dataset_mlb.joblib'
    loaded_model = joblib.load(model_file_path)

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
    
    df_students['Recommendation'] = recommendations_list
    
    df_students.to_excel(Test_excel_filename, sheet_name=course_id, index=False)
            
    messagebox.showinfo("Recommendation", f"Recommendation Complete for Course ID {course_id}")
    
    
#========================================================================================================================== 
#Class Analysis

def class_recommendation(l):   
    
    cls_id,course_id = l[0],l[1]

    generate_data(course_id)
    training_data(course_id)
    dataprep(course_id)
    
    df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
    df_students = pd.read_excel(Test_excel_filename,sheet_name=course_id)
    df_students = df_students[df_students['Class'] == cls_id]
    df_students_check = df_students['Class']
    Assessments = list(df_courses['Assessments'])
    stu_marks_avg = []
    
    if cls_id in list(df_students_check):
        Converted_Assessments_name = []
    
        for i,j in enumerate(Assessments):
            converted_column_name = j + ' Converted'
            Converted_Assessments_name.append(converted_column_name)
        
        for assessment in Converted_Assessments_name:
            average = df_students[assessment].mean() 
            stu_marks_avg.append(average)
        
        model_file_path = course_id + "_dataset_model.joblib"
        mlb_file_path = course_id + '_dataset_mlb.joblib'
        loaded_model = joblib.load(model_file_path)

        test_data = {assessment: [f"{mark:.2f}"] for assessment, mark in zip(Assessments, stu_marks_avg)}

        test_df = pd.DataFrame(test_data)

        predictions = loaded_model.predict(test_df)
        mlb = joblib.load(mlb_file_path) 
        predicted_labels = mlb.inverse_transform(predictions.toarray())

        recommendation = ""
        for i, labels in enumerate(predicted_labels):
            if labels:
                recommendation += f"{', '.join(labels)}; "
            else:
                recommendation += "No Recommendations; "
                
        formatted_marks = "\n".join([f"{assessment}: {f'{float(mark[0]):.2f}'}" for assessment, mark in test_data.items()])
        messagebox.showinfo("Recommendation", f"Class: {cls_id}\nCourseID: {course_id}\n\n{formatted_marks}\n\nRecommendation: {recommendation}")
    else:
        messagebox.showerror("No class", f"Class: {cls_id} is not available")
    
    
#========================================================================================================================== 
#Find students

def findthem(student_id,class_id,course_id):
    
    df = pd.read_excel('Students.xlsx', sheet_name=course_id)
    df_courses = pd.read_excel('Courses.xlsx',sheet_name=course_id)
    
    result = df[(df['Class'] == class_id) & (df['Student Id'] == int(student_id))]
    Assessments = list(df_courses['Assessments'])
    
    Converted_Assessments_name = []

    for i,j in enumerate(Assessments):
        converted_column_name = j + ' Converted'
        Converted_Assessments_name.append(converted_column_name)

    stu_marks = []
    for assessment in Converted_Assessments_name:
        stu_marks.append(result[assessment].values[0])
        
    test_data = {assessment: [f"{mark}"] for assessment, mark in zip(Assessments, stu_marks)}
    
    recommendation = result['Recommendation'].values[0]
    formatted_marks = "\n".join([f"{assessment}: {mark[0]}" for assessment, mark in test_data.items()])
    messagebox.showinfo("Recommendation", f"Student ID: {student_id}\nClass: {class_id}\nCourseID: {course_id}\n\n{formatted_marks}\nTotal: {result['Total'].values[0]}\n\nRecommendation: {recommendation}")
    

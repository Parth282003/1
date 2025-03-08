#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load and clean the dataset

data = pd.read_csv('student-mat.csv', delimiter=';', quotechar='"')

# Extract relevant columns
data_cleaned = data[['studytime', 'absences', 'G3']].copy()

# Map studytime categories to approximate weekly study hours
studytime_mapping = {1: 2.5, 2: 5, 3: 7.5, 4: 10}
data_cleaned['Study Hours'] = data_cleaned['studytime'].map(studytime_mapping)

# Convert absences to attendance percentage (assuming 200 total classes)
data_cleaned['Attendance'] = ((200 - data_cleaned['absences'].astype(float)) / 200) * 100

# Create 'Pass' column (G3 >= 10 considered a pass)
data_cleaned['Pass'] = (data_cleaned['G3'].astype(int) >= 10).astype(int)

# Final dataset
data_cleaned = data_cleaned[['Study Hours', 'Attendance', 'Pass']]

# Split data into training and testing sets
X = data_cleaned[['Study Hours', 'Attendance']]
y = data_cleaned['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay(conf_matrix, display_labels=['Fail (0)', 'Pass (1)']).plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





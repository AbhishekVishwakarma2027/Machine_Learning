# QSkill Internship – Artificial Intelligence & Machine Learning

## Internship Details
- **Organization:** QSkill  
- **Domain:** Artificial Intelligence & Machine Learning  
- **Internship Duration:** 10 January 2026 – 10 February 2026  
- **Task Submission Deadline:** 10 February 2026  

As part of the QSkill Internship program, I completed **two machine learning tasks** focusing on **classification** and **regression** problems using Python and standard ML libraries.

---

## Task 1: Spam Mail Detector

### Objective
To build a machine learning model that classifies messages as **Spam** or **Non-Spam (Ham)** using textual data.

### Dataset
- **SMS Spam Collection Dataset** (UCI Machine Learning Repository)  
- Contains labeled SMS messages as `spam` or `ham`.

### Methodology
1. Loaded the dataset containing text messages and labels  
2. Performed text preprocessing:
   - Converted text to lowercase  
   - Removed stopwords  
   - Tokenized text  
3. Converted text data into numerical features using:
   - Bag of Words / TF-IDF Vectorization  
4. Split the dataset into training and testing sets  
5. Trained machine learning models:
   - Naive Bayes  
   - Logistic Regression  
6. Evaluated model performance using:
   - Accuracy  
   - Precision  
   - F1-Score  

### Outcome
The trained model successfully classified spam and non-spam messages with good accuracy and demonstrated effective use of basic Natural Language Processing techniques.

### Skills Gained
- Text preprocessing  
- Feature extraction  
- Natural Language Processing (NLP)  
- Classification algorithms  
- Model evaluation metrics  

---

## Task 2: House Price Prediction

### Objective
To predict house prices based on various numerical features such as size, location, and number of bedrooms.

### Dataset
- **Boston Housing Dataset** (or similar structured housing dataset)

### Methodology
1. Loaded and explored the dataset  
2. Analyzed data distributions and feature relationships  
3. Handled missing values (if present)  
4. Normalized numerical features  
5. Split the dataset into training and testing sets  
6. Trained a regression model:
   - Linear Regression  
7. Evaluated model performance using:
   - Mean Squared Error (MSE)  

### Outcome
The regression model was able to predict house prices effectively, demonstrating the application of supervised learning for numerical prediction tasks.

### Skills Gained
- Tabular data analysis  
- Data preprocessing  
- Regression modeling  
- Feature engineering  
- Model evaluation using error metrics  

---

## Tools & Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## Project Structure
```
Machine_Learning/
│
├──dataset\
|  |──train.csv
|  |──SMSSpamCollection.data
│
|──House_Price_Prediction.ipynb
|──SMS_Detector.ipynb
└── README.md
```

---

## Learning Outcome
Through this internship, I gained hands-on experience in:
- End-to-end machine learning workflow  
- Text-based classification problems  
- Regression problems on structured data  
- Data preprocessing and feature engineering  
- Model evaluation and performance analysis  

---

## Author
**Abhishek Kumar Vishwakarma**  
B.Tech Computer Science Engineering  
(Data Science & Artificial Intelligence)

---

## Acknowledgement
I would like to thank **QSkill** for providing this internship opportunity and hands-on exposure to real-world machine learning applications.

# Fake_News_Detection

Project Title: Fake News Detection System

Description:
This project implements a Fake News Detection System using Logistic Regression and TF-IDF Vectorization for classifying news articles into real or fake categories. The system preprocesses data using techniques such as stemming and missing value handling and achieves 95% accuracy.

Technologies Used:
	•	Python
	•	scikit-learn for model training and evaluation
	•	NLTK for text preprocessing
	•	Pandas for data manipulation

Steps:
	1.	Data Preprocessing:
	•	Handle missing values
	•	Combine text features: author + title
	•	Apply Porter Stemmer for stemming
	2.	Text Vectorization:
	•	Use TF-IDF Vectorizer to convert text data into numerical features
	3.	Model Training:
	•	Train using Logistic Regression
	4.	Model Evaluation:
	•	Achieved 95% accuracy on the test set

How to Use:
	1.	Clone the repository.
	2.	Install required libraries
	3.	Run fake_news.py to train the model and make predictions.

Future Scope:
	•	The system can be scaled for large datasets and deployed for real-time fake news detection on digital platforms.

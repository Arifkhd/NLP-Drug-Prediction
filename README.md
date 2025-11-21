🧠 NLP Disease Prediction<br>
📌 Project Overview<br>
<br>
This project predicts medical conditions (such as Depression, High Blood Pressure, and Type 2 Diabetes) based on patient drug reviews using Natural Language Processing (NLP) and machine learning.<br>
<br>
The goal is to analyze patient-written text, extract meaningful features, and classify the disease accurately based on review patterns.<br>

| Column Name    | Description                        |
| -------------- | ---------------------------------- |
| `review_text`  | Patient-written medical review     |
| `condition`    | Disease/medical condition (target) |
| `drug_name`    | Drug being used                    |
| `rating`       | User-given rating                  |
| `useful_count` | Votes indicating review usefulness |
<br>

🎯 Objectives
<br>
1) Convert patient-written text into structured machine learning features<br>
<br>
2) Build NLP-based classification models
<br>
3) Evaluate model performance using classification metrics
<br>
4) Deploy the prediction system using Streamlit
<br>

🧠 Approach & Methodology<br>
1️⃣ Text Preprocessing
<br>
Lowercasing
Removing stopwords<br>
Removing punctuation<br>
Tokenization<br>
Lemmatization/Stemming,br>
<br>
2️⃣ Feature Engineering
<br>
TF-IDF Vectorization<br>
Bag-of-Words (BoW)<br>
N-gram analysis,br>
WordCloud for most common terms<br>
<br>
3️⃣ Model Building
<br>
Models tested:<br>
Logistic Regression<br>
Random Forest<br>
SVM<br>
Multinomial Naïve Bayes<br>
Fine-tuned using hyperparameter tuning<br>
<br>
4️⃣ Evaluation Metrics
<br>
Accuracy<br>
Precision<br>
Recall<br>
F1-Score<br>
Confusion Matrix<br>
<br>

🧪 Challenges Faced During Development<br>
📍 Class Imbalance
<br>
Some diseases had significantly fewer samples, causing biased model predictions.<br>
Handled using SMOTE and class weighting.<br>
<br>
📍 Noisy and Unstructured Text
<br>
Patient reviews contained grammatical errors, misspellings, and informal writing styles.<br>
Advanced text preprocessing helped improve model clarity.<br>
<br>
📍 Similar Language Between Diseases<br>
<br>
Several conditions had overlapping vocabulary, making classification difficult.<br>
Feature engineering and model tuning improved separation.<br>
<br>
🏆 Results
<br>
Best model achieved significant improvement in accuracy and F1-score after tuning.<br>

TF-IDF + Logistic Regression performed strongly for textual classification.<br>

Visualizations showed clear keyword patterns for different diseases.<br>

<br>
💻 Tech Stack<br>
Languages<br>
Python<br>
Libraries<br>
Pandas<br>
NumPy<br>
Scikit-learn<br>
NLTK<br>
Matplotlib<br>
Seaborn<br>
Streamlit<br>
<br>
🚀 How to Run<br>
1️⃣ Clone the repository<br>
git clone https://github.com/<username>/nlp-disease-prediction<br>
<br>
2️⃣ Install dependencies,br>
pip install -r requirements.txt,br>

3️⃣ Run the Streamlit App<br>
streamlit run app.py<br>

4️⃣ Access the Application<br>

Open your browser at:<br>

http://localhost:8501<br>
<br>

📈 Future Improvements
<br>
Deploy as a REST API using FastAPI/Flask
<br>
Train with transformer-based models (BERT, RoBERTa, DistilBERT)
<br>
Integrate medical knowledge graphs
<br>
Add model drift monitoring and MLOps features
<br>
<br>
🤝 Contribution<br>
Contributions are welcome — please open an issue before making changes.

# 🚨 **Insurance Claims Fraud Detection Model**  

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/) [![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)](https://xgboost.readthedocs.io/en/latest/) [![NLP](https://img.shields.io/badge/NLP-NLTK-orange)](https://www.nltk.org/)  

---

## 📌 **Project Overview**  
This project focuses on detecting **fraudulent insurance claims** by leveraging **Machine Learning** and **Natural Language Processing (NLP)** techniques. The dataset consists of **structured numerical data** and **unstructured claim descriptions**, enabling a **hybrid fraud detection approach** that enhances predictive accuracy.  

💡 **Key Objectives:**  

✔ Identify **patterns in fraudulent insurance claims**  
✔ Develop an **ML model** optimized for fraud detection  
✔ Utilize **NLP** for text-based anomaly detection  
✔ Improve fraud detection **accuracy while reducing false positives**  
✔ Create **real-time fraud risk visualizations** in **Power BI**  

---

## 📊 **Dataset & Methodology**  
### **Dataset:**  
The dataset includes **structured claim details** (policy info, claim amount, etc.) and **unstructured descriptions** of incidents. It has **multiple categorical, numerical, and text-based features**.  

### **Key Features Considered:**  
| **Feature** | **Description** |
|------------|----------------|
| `fraud_reported` | Target variable - indicates if a claim is fraudulent |
| `incident_type` | Type of claim incident (Theft, Collision, etc.) |
| `insured_sex` | Gender of the insured person |
| `claim_amount` | Total claimed amount |
| `incident_description` | Unstructured text describing the incident |

### **Data Preprocessing & Feature Engineering:**  
✔ **Handled missing values** and performed **categorical encoding**  
✔ **Cleaned and tokenized claim descriptions** using **NLTK**  
✔ **Applied TF-IDF vectorization** to extract text insights  
✔ **Standardized numerical features** for model training  

---

## 📁 **Project Structure**  
```
📂 data/                # Raw and cleaned datasets
📂 notebooks/           # Jupyter Notebooks for EDA, ML modeling, and analysis
📂 src/                 # Python scripts for modularization
📂 results/             # Model results and performance metrics
📂 config/              # Configuration settings for hyperparameters
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── fraud_detection.py # Main fraud detection model script
├── dashboard.pbix     # Power BI fraud visualization dashboard
```

---

## ⚙️ **Machine Learning Models Used**
We experimented with **multiple ML models**, and **XGBoost** achieved the best performance.

| **Model**                  | **Accuracy** | **False Positive Reduction** |
|----------------------------|-------------|------------------------------|
| **Logistic Regression**    | 78%         | 10%                          |
| **Random Forest**          | 83%         | 18%                          |
| **XGBoost (Optimized)**    | **87%**     | **30%**                       |

📌 **Best Model**: **XGBoost**, achieving **87% accuracy** and a **30% reduction in false positives**, making it the most effective for fraud detection.

---

## 📈 **Key Insights & Findings**
✔ Fraudulent claims often contain **specific keywords** in descriptions (e.g., "stolen," "urgent").  
✔ **Claim amount thresholds** can be strong indicators of fraud risk.  
✔ **XGBoost and NLP techniques combined** improve **fraud detection accuracy significantly**.  

---

## 🚀 **Business Impact**
📊 **Fraud Prevention**: Helps insurers **identify fraudulent claims**, preventing financial losses.  
📉 **False Positive Reduction**: Improves fraud detection while minimizing **legitimate claim rejections**.  
🧠 **Data-Driven Decision Making**: Provides **underwriters** with actionable **risk insights**.  

---

## 🖥️ **Technologies Used**
| **Category**         | **Technologies** |
|----------------------|-----------------|
| **Languages**        | Python, SQL |
| **Data Processing**  | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-learn, XGBoost, TensorFlow |
| **Natural Language Processing (NLP)** | NLTK, TF-IDF |
| **Data Visualization** | Matplotlib, Seaborn, Power BI |
| **Deployment**       | Flask, Docker, AWS Lambda |

---

## 📚 **References**
- 📄 Kaggle Dataset: [Insurance Fraud Claims Dataset](https://www.kaggle.com/)  
- 📄 Scikit-learn Documentation: [Machine Learning in Python](https://scikit-learn.org/stable/)  
- 📄 Power BI Dashboards: [Data Visualization](https://powerbi.microsoft.com/)  
- 📄 Fraud Detection Research: [IEEE Papers](https://ieeexplore.ieee.org/)  

---

## 🤝 **Contributors**
👤 **Nava Bhargav Gedda**  
📩 [navabhargavg@gmail.com](mailto:navabhargavg@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/nava-bhargav-gedda-4a4a30151) | 🌐 [GitHub](https://github.com/Navabhargav)  

---

## ⭐ **Like this Project?**
If you found this project useful, **give it a star ⭐** on GitHub and share it with others! 🚀

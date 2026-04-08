# 🤖 Machine Learning using Python

This repository contains basic concepts, notes, and implementations of **Machine Learning using Python**. It is designed for beginners to understand core ML concepts along with practical examples.

---

## 📚 Contents

- Introduction to Machine Learning
- Why Machine Learning?
- Why Python for ML?
- Types of Machine Learning:
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Representation of Data (Features & Samples)
- Essential Python Libraries:
  - NumPy
  - SciPy
  - pandas
  - matplotlib
- Tools:
  - Jupyter Notebook
  - scikit-learn

---

## 🚀 Applications of Machine Learning

- Movie and product recommendations  
- Face detection and image recognition  
- Spam email classification  
- Fraud detection in transactions  
- Medical diagnosis  

---

## 🧠 Machine Learning Concepts

### 🔹 Supervised Learning
- Works with labeled data  
- Used for:
  - Classification
  - Regression  

### 🔹 Unsupervised Learning
- Works with unlabeled data  
- Finds hidden patterns  
- Example: Clustering  

### 🔹 Reinforcement Learning
- Learns through feedback (rewards & penalties)  

---

## 📊 Example Project: Iris Classification

- Dataset: Iris dataset from `sklearn`
- Problem Type: Classification
- Classes:
  - Setosa
  - Versicolor
  - Virginica  

### Steps Involved:
1. Load dataset  
2. Split into training and testing data  
3. Train model  
4. Make predictions  
5. Evaluate accuracy  

---

## 🔍 Algorithm Used: K-Nearest Neighbors (KNN)

### How it Works:
1. Choose value of K  
2. Calculate distance from neighbors  
3. Select nearest K points  
4. Assign class based on majority  

### ✅ Advantages:
- Simple to implement  
- Works well with large datasets  
- No training phase (lazy learner)  

### ❌ Disadvantages:
- High computation cost  
- Choosing K can be tricky  

---

## 💻 Sample Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load dataset
iris = load_iris()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0
)

# Create model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Prediction
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)

# Evaluation
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)

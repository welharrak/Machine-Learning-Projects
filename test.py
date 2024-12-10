import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Title
st.title("Interactive Machine Learning Classifier")

# Model Selection
model_choice = st.selectbox("Choose a Model", ["k-NN", "SVM"])
num_points = st.slider("Number of Points", 50, 500, step=50)

# Generate Data
X, y = make_classification(n_samples=num_points, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Model Training
if model_choice == "k-NN":
    k = st.slider("k (for k-NN)", 1, 10, step=1)
    model = KNeighborsClassifier(n_neighbors=k)
elif model_choice == "SVM":
    model = SVC(kernel='linear')

model.fit(X, y)
y_pred = model.predict(X)

# Visualization
fig, ax = plt.subplots()
for label in np.unique(y_pred):
    ax.scatter(X[y_pred == label, 0], X[y_pred == label, 1], label=f"Class {label}")

plt.legend()
st.pyplot(fig)

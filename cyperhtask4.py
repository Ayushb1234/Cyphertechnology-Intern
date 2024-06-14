import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load data
data = pd.read_csv('spam.csv')
data.columns = ['label', 'text']

# Preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Splitting data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Model prediction
y_pred = model.predict(X_test_vec)
y_probas = model.predict_proba(X_test_vec)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_probas)
roc_auc = auc(fpr, tpr)

# Streamlit app
st.title('Spam Detection App')

st.write(f'Model Accuracy: {accuracy:.2f}')

user_input = st.text_area("Enter a message to check if it's spam or not:")

if st.button('Predict'):
    user_input_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vec)
    if prediction == 1:
        st.write('This is a spam message.')
    else:
        st.write('This is not a spam message.')

# Confusion Matrix Heatmap (Seaborn)
st.write('Confusion Matrix:')
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
st.pyplot(fig)

# ROC Curve (Matplotlib)
st.write('ROC Curve:')
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)

# Interactive ROC Curve (Plotly)
st.write('Interactive ROC Curve:')
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
fig.update_layout(title='Receiver Operating Characteristic', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig)

# Interactive Confusion Matrix Heatmap (Plotly)
st.write('Interactive Confusion Matrix Heatmap:')
z = cm
x = ['Ham', 'Spam']
y = ['Ham', 'Spam']
fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
fig.update_layout(title='Confusion Matrix Heatmap')
st.plotly_chart(fig)

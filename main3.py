import os
import glob
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
import streamlit as st

# Global variables
happy_dir = 'happy'
sad_dir = 'sad'
dataset_file = 'dataset.npy'
model_file = 'model.joblib'

# Calculate GLCM features
def calculate_glcm_features(img):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    glcm = graycomatrix(img, [1], angles, 256, symmetric=True, normed=True)
    
    features = []
    
    for prop in properties:
        feature = graycoprops(glcm, prop).ravel()
        features.append(feature)
        
        st.write(f"{prop}: {feature}")
    
    return np.concatenate(features)

# Train the model
def train_model(k):
    st.write("Training Model...")
    
    # Load image files and labels
    happy_files = glob.glob(os.path.join(happy_dir, '*'))
    sad_files = glob.glob(os.path.join(sad_dir, '*'))
    files = happy_files + sad_files
    labels = [0] * len(happy_files) + [1] * len(sad_files)  # 0 for happy, 1 for sad
    
    # Preprocess images and extract GLCM features
    features = [calculate_glcm_features(cv2.resize(cv2.imread(file, 0), (48, 48))) for file in files]

    # Ensure features and labels are of the same size
    features, labels = np.array(features), np.array(labels)
    assert features.shape[0] == labels.shape[0], "Mismatch in features and labels sizes"

    # Train KNN classifier
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    
    st.write(f"Model accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1_score:.2f}")
    
    # Save the model
    joblib.dump(model, model_file)
    st.write("Model saved!")
    
    # Save the dataset
    dataset = np.column_stack((features, labels))
    np.save(dataset_file, dataset)
    st.write("Dataset saved!")

# Streamlit UI - Home Page
def home():
    st.title('Happy or Sad Image Classifier')
    
    # Check if model and dataset files exist
    model_exists = os.path.exists(model_file)
    dataset_exists = os.path.exists(dataset_file)
    
    # Add "Train Model" button
    if st.button("Train Model"):
        st.session_state.page = 'Train'

    # Image classification
    file = st.file_uploader("Upload an image")
    col1, col2 = st.columns(2)
    if file is not None:
        # Preprocess and extract GLCM features
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        features = calculate_glcm_features(img)
        
        col1.image(img, caption="Uploaded Image", use_column_width=False)

        if model_exists:
            # Load the model and predict
            model = joblib.load(model_file)
            prediction = model.predict([features])[0]
            if prediction == 0:
                col2.write("The image is happy!")
            else:
                col2.write("The image is sad!")
                
            # Calculate metrics
            if dataset_exists:
                dataset = np.load(dataset_file)
                X_test, y_test = dataset[:, :-1], dataset[:, -1]
                y_pred = model.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred)
                recall = metrics.recall_score(y_test, y_pred)
                f1_score = metrics.f1_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy*100:.2f}%")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1-Score: {f1_score:.2f}")

# Streamlit UI - Train Page
def train():
    st.title('Train Model')
    
    # Add "Back" button
    if st.button("Back"):
        st.session_state.page = 'Home'
    
    # Select K value
    k = st.selectbox("Select K value", [1, 3, 5, 7, 9])
    
    # Train the model
    if st.button("Train", key="train_button"):
        train_model(k)

# Streamlit App
def main():
    # Initialize Streamlit session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    
    # Page navigation
    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Train':
        train()

if __name__ == "__main__":
    main()

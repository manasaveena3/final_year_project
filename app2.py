# app2.py
from huggingface_hub import hf_hub_download
import streamlit as st
import json
import hashlib
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from transformers import ViTModel
from PIL import Image
import pickle
import gdown
import os

# ------------------ AUTH ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# ------------------ MODEL CLASS ------------------
class HybridViTResNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet_fc = nn.Linear(2048, 512)

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit_fc = nn.Linear(768, 512)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        resnet_features = self.resnet(x).view(x.size(0), -1)
        resnet_features = self.resnet_fc(resnet_features)

        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        vit_features = self.vit_fc(vit_features)

        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        return self.fc(combined_features)

# ------------------ DOWNLOAD MODEL ------------------
def download_model_if_needed():
    file_id = "15uX6U-dwmiCo-noPrnv51pzzkGmm4lXq"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "hybrid_vit_resnet2.pth"

    if not os.path.exists(output):
        with st.spinner("🔄 Downloading model weights from Google Drive..."):
            gdown.download(url, output, quiet=False)

# ------------------ LOGIN PAGE ------------------
def login_page():
    st.title("🔐 Login or Register")
    users = load_users()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state["user"] = username
                st.success("✅ Logged in!")
            else:
                st.error("❌ Invalid credentials.")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if new_user in users:
                st.warning("⚠️ Username already exists!")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.success("✅ User registered! Please login.")

# ------------------ NUMERICAL PREDICTION ------------------
def numerical_prediction(log_model):
    st.header("🧪 Step 1: Enter Patient Details")
    with st.form("numerical_form"):
        col1, col2, col3 = st.columns(3)
        Age = col1.number_input("Age")
        Sex = col2.selectbox("Sex", ["Male", "Female"])
        ALB = col3.number_input("Albumin (ALB)")
        ALP = col1.number_input("Alkaline Phosphatase (ALP)")
        ALT = col2.number_input("Alanine Transaminase (ALT)")
        AST = col3.number_input("Aspartate Transaminase (AST)")
        BIL = col1.number_input("Bilirubin (BIL)")
        CHE = col2.number_input("Cholinesterase (CHE)")
        CHOL = col3.number_input("Cholesterol (CHOL)")
        CREA = col1.number_input("Creatinine (CREA)")
        GGT = col2.number_input("GGT")
        PROT = col3.number_input("Protein (PROT)")

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        sex_binary = 1 if Sex == "Male" else 0
        X = pd.DataFrame([[Age, sex_binary, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT]],
                         columns=["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"])
        try:
            pred = log_model.predict(X)[0]
            st.success(f"🧾 Prediction: {'🟢 YES - Proceed to Imaging' if pred == 1 else '🔴 NO - Not Required'}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------ IMAGE CLASSIFICATION ------------------
def image_classification(model, device):
    st.header("🖼️ Step 2: Upload Liver CT Scan Image")
    uploaded_img = st.file_uploader("Upload an Image (.jpg/.png)", type=["jpg", "png"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", width=300)

        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        with st.spinner("🔬 Classifying..."):
            img_tensor = img_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            class_names = [" No Fibrosis", "Portal Fibrosis", "Periportal Fibrosis", "Septal Fibrosis", " Cirrhosis"]
            st.success(f"🧠 Predicted Class: **{class_names[pred_class]}**")

# ------------------ MAIN ------------------
def main():
    st.set_page_config(page_title="Liver Condition App", layout="wide")

    if "user" not in st.session_state:
        login_page()
        return

    # Download model if needed
    download_model_if_needed()

    # Load image classification model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridViTResNet(num_classes=5).to(device)
    repository_id = "Nithin-nani/1st_model"  # Replace with "Nithin-nani/1st_model" if that's correct
    filename = "hybrid_vit_resnet2.pth" 
    
    model.load_state_dict(torch.load(hf_hub_download(repo_id=repository_id, filename=filename), map_location=device))
    model.eval()

    # Load logistic regression model from local
    try:
        with open("log_reg_model983.pkl", "rb") as f:
            log_model = pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Logistic Regression model file not found. Make sure 'log_reg_model983.pkl' is in the same folder.")
        return

    # Sidebar menu
    menu = st.sidebar.selectbox("📋 Menu", ["Numerical Prediction", "Image Classification", "Logout"])

    if menu == "Numerical Prediction":
        numerical_prediction(log_model)
    elif menu == "Image Classification":
        image_classification(model, device)
    elif menu == "Logout":
        del st.session_state["user"]
        st.rerun()

if __name__ == "__main__":
    main()

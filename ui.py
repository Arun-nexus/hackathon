import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import datetime
import os
import requests
from PIL import Image
import plotly.express as px

MODEL_PATH = r"new_cnn_model.pth"
DETAILS_FILE = "details.xlsx"
NUTRITION_FILE = "nutrition.xlsx"
API_ID = "f3ea175a"
API_KEY = "eaa480a05e96324249b5ad419fd89288"

class cnn_model(nn.Module):
    def __init__(self, num_classes=101):
        super(cnn_model, self).__init__()
        self.base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        for param in self.base.features.parameters():
            param.requires_grad = False

        infeatures = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(infeatures, 512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base(x)


@st.cache_resource
def load_model():
    model = cnn_model(num_classes=101)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.eval()
            st.success("✅ MobileNetV3 model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None
    else:
        st.error("❌ Model file not found. Check path.")
        return None


def ensure_files():
    if not os.path.exists(DETAILS_FILE):
            st.subheader("Enter Your Details")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=1, max_value=120)
            height = st.number_input("Height (cm)", min_value=50, max_value=250)
            weight = st.number_input("Weight (kg)", min_value=10, max_value=300)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            if st.button("Save Details"):
                df = pd.DataFrame([{
                    "name": name,
                    "age": age,
                    "height": height,
                    "weight": weight,
                    "gender": gender
                }])
                df.to_excel(DETAILS_FILE, index=False)
                st.success("Details saved!")

    if not os.path.exists(NUTRITION_FILE):
        today = datetime.date.today().strftime("%Y-%m-%d")
        pd.DataFrame(
            [{"date": today, "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "water": 0}]
        ).to_excel(NUTRITION_FILE, index=False)


def query_nutritionix(food):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {"x-app-id": API_ID, "x-app-key": API_KEY, "Content-Type": "application/json"}
    try:
        res = requests.post(url, json={"query": food}, headers=headers, timeout=8)
        if res.status_code == 200:
            return res.json()["foods"][0]
    except Exception:
        return None
    return None


def predict_image(model, image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(np.expand_dims(arr, 0))
    with torch.no_grad():
        out = model(t)
        probs = F.softmax(out, dim=1)[0].numpy()
    pred = np.argmax(probs)
    return f"class_{pred}"


def update_nutrition(calories, protein, carbs, fat, water):
    df = pd.read_excel(NUTRITION_FILE)
    today = datetime.date.today().strftime("%Y-%m-%d")
    if today not in df["date"].values:
        df = pd.concat(
            [pd.DataFrame([{"date": today, "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "water": 0}]), df],
            ignore_index=True,
        )
    idx = df[df["date"] == today].index[0]
    df.loc[idx, ["calories", "protein", "carbs", "fat", "water"]] = [
        df.loc[idx, "calories"] + calories,
        df.loc[idx, "protein"] + protein,
        df.loc[idx, "carbs"] + carbs,
        df.loc[idx, "fat"] + fat,
        df.loc[idx, "water"] + water,
    ]
    df.to_excel(NUTRITION_FILE, index=False)
    return df


st.set_page_config(page_title="🍎 AI Nutrition Tracker", layout="wide")
st.title("🥗 AI-Based Nutrition Tracker (MobileNetV3 CNN Model)")

ensure_files()
model = load_model()

details = pd.read_excel(DETAILS_FILE).iloc[0].to_dict()
df = pd.read_excel(NUTRITION_FILE)
today = datetime.date.today().strftime("%Y-%m-%d")

st.sidebar.header("👤 User Details")
for k, v in details.items():
    st.sidebar.write(f"**{k.capitalize()}**: {v}")

st.sidebar.divider()
if st.sidebar.button("💧 Add 250ml Water"):
    df = update_nutrition(0, 0, 0, 0, 0.25)
    st.sidebar.success("Water updated!")

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.header("🍴 Add Food")
    option = st.radio("Select Input Method", ["Upload Image", "Camera", "Manual"], horizontal=True)

    if option == "Upload Image":
        uploaded = st.file_uploader("Upload food image", type=["jpg", "png", "jpeg"])
        if uploaded and model:
            img = Image.open(uploaded)
            st.image(img, width=250)
            pred = predict_image(model, img)
            st.success(f"Predicted: {pred}")
            data = query_nutritionix(pred)
            if data:
                st.write(data)
                df = update_nutrition(
                    data.get("nf_calories", 0),
                    data.get("nf_protein", 0),
                    data.get("nf_total_carbohydrate", 0),
                    data.get("nf_total_fat", 0),
                    0,
                )
                st.success("✅ Nutrition updated!")

    elif option == "Manual":
        food = st.text_input("Enter food (e.g., 2 eggs, 100g rice)")
        if st.button("Fetch Nutrition") and food:
            data = query_nutritionix(food)
            if data:
                st.write(data)
                df = update_nutrition(
                    data.get("nf_calories", 0),
                    data.get("nf_protein", 0),
                    data.get("nf_total_carbohydrate", 0),
                    data.get("nf_total_fat", 0),
                    0,
                )
                st.success("✅ Nutrition updated!")

    elif option == "Camera":
        camera = st.camera_input("Take a photo")
        if camera and model:
            img = Image.open(camera)
            pred = predict_image(model, img)
            st.success(f"Predicted: {pred}")
            data = query_nutritionix(pred)
            if data:
                st.write(data)
                df = update_nutrition(
                    data.get("nf_calories", 0),
                    data.get("nf_protein", 0),
                    data.get("nf_total_carbohydrate", 0),
                    data.get("nf_total_fat", 0),
                    0,
                )
                st.success("✅ Nutrition updated!")

with col2:
    st.header("📊 Today's Progress")
    today_data = df[df["date"] == today].iloc[0]
    chart_df = pd.DataFrame(
        {
            "Nutrient": ["Calories", "Protein", "Carbs", "Fat", "Water"],
            "Value": [
                today_data["calories"],
                today_data["protein"],
                today_data["carbs"],
                today_data["fat"],
                today_data["water"],
            ],
        }
    )
    fig = px.bar(chart_df, x="Nutrient", y="Value", title="Today's Nutrient Intake", color="Nutrient")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Made with ❤️ by Arun's MobileNetV3 CNN AI")

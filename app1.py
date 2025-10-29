import streamlit as st
import faiss
import torch
import open_clip
from PIL import Image
import numpy as np
import os

# --- 1️⃣ モデル準備 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

# --- 2️⃣ cards フォルダから画像を読み込む ---
image_folder = "cards"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('jpg','png','jpeg'))]
if len(image_paths) == 0:
    st.error("⚠️ 'cards' フォルダに画像がありません。")
else:
    st.success(f"{len(image_paths)} 枚のトレカ画像を読み込みました。")

# --- 3️⃣ 特徴量を抽出してデータベース作成 ---
@st.cache_resource
def build_index():
    features = []
    for p in image_paths:
        img = preprocess(Image.open(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
    features = np.vstack(features)

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    return index, features

index, features = build_index()

# --- 4️⃣ 検索UI ---
st.title("📸 ホンジュントレカ検索サイト")
st.write("アップロードした写真がどのトレカに最も近いかをAIが判定します。")

uploaded = st.file_uploader("トレカの写真をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded:
    query_img = Image.open(uploaded)
    st.image(query_img, caption="アップロード画像", width=250)

    # 特徴量を抽出
    image = preprocess(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_feat = model.encode_image(image)
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

    # 類似検索
    D, I = index.search(q_feat.cpu().numpy(), 3)
    st.subheader("🔍 類似トレカ候補")
    for rank, idx in enumerate(I[0]):
        match_path = image_paths[idx]
        st.image(match_path, caption=f"候補 {rank+1}: {os.path.basename(match_path)} (類似度 {D[0][rank]:.2f})", width=250)

import streamlit as st
import faiss
import torch
import open_clip
from PIL import Image
import numpy as np
import os

# --- 1️⃣ モデル準備 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.to(device)
model.eval()

# --- 2️⃣ cards フォルダから画像を読み込む ---
image_folder = "cards"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('jpg','png','jpeg'))]
if len(image_paths) == 0:
    st.error("⚠️ 'cards' フォルダに画像がありません。")
else:
    st.success(f"{len(image_paths)} 枚のトレカ画像を読み込みました。")

# --- 3️⃣ 特徴量を抽出してデータベース作成 ---
@st.cache_resource
def build_index():
    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy().astype("float32"))
    features = np.vstack(features)

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    return index, features

index, features = build_index()

# --- 4️⃣ 検索UI ---
st.title("📸 Hongjoongトレカ検索")
st.write("アップロード画像と最も類似するトレカを検索します。")

uploaded = st.file_uploader("トレカ写真をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="アップロード画像", width=300)

    # 特徴量抽出
    image_tensor = preprocess(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_feat = model.encode_image(image_tensor)
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

    q_feat = q_feat.cpu().numpy().astype("float32")

    # 類似検索（コサイン類似度）
    D, I = index.search(q_feat, k=min(5, len(image_paths)))

    st.subheader("🔍 類似トレカ候補")

    for rank, idx in enumerate(I[0]):
        match_path = image_paths[idx]
        similarity = D[0][rank]
    
    # --- 💡 ファイル名からカード名を整形 ---
    filename = os.path.basename(match_path)
    card_name = os.path.splitext(filename)[0]  # 拡張子除去
    card_name = card_name.replace("_", " ").replace("-", " ").title()  # 見やすく

    # --- 📸 表示 ---
    st.image(
        match_path,
        caption=f"候補 {rank+1}: {card_name}\n類似度 {similarity:.3f}",
        width=300,
    )

# --- 🏁 一致度メッセージ ---
best_score = float([0][0])
best_card = os.path.splitext(os.path.basename(image_paths[[0][0]]))[0]

if best_score > 0.98:
    st.success(f"✅ 最も一致したカード: {best_card}（類似度 {best_score:.3f}）")
elif best_score > 0.8:
    st.info(f"🟩 類似カード: {best_card}（類似度 {best_score:.3f}）")
else:
    st.warning(f"⚠️ 一致度が低い（{best_score:.3f}） → 異なるカードの可能性あり")

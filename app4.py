import streamlit as st
import faiss
import torch
import open_clip
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import cv2, os

# --- モデル準備 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.to(device)
model.eval()

# --- カード情報の読み込み ---
image_folder = "cards"
info_path = os.path.join(image_folder, "info.csv")
if os.path.exists(info_path):
    info_df = pd.read_csv(info_path)
else:
    info_df = pd.DataFrame(columns=["filename", "name", "series", "number"])

# --- ファイルパス一覧 ---
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(("jpg", "jpeg", "png"))
]

# --- OpenCV前処理（枠検出＋白背景） ---
def preprocess_card(image: Image.Image, size=512):
    img = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = img_cv[y:y+h, x:x+w]
    else:
        cropped = img_cv

    pil_cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_cropped = _center_square_white(pil_cropped, size)
    return pil_cropped

def _center_square_white(image: Image.Image, size=512):
    img = image.convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    left, top = (w - min_side) // 2, (h - min_side) // 2
    right, bottom = left + min_side, top + min_side
    img = img.crop((left, top, right, bottom))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img)
    img = ImageOps.contain(bg, (size, size))
    return img

# --- 特徴量DBを構築 ---
@st.cache_resource
def build_index():
    features = []
    for path in image_paths:
        img = preprocess_card(Image.open(path))
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

# --- Streamlit UI ---
st.title("📸 Hongjoong トレカ識別アプリ")
st.write("AIで画像検索＋カード名検索ができます。")

tab1, tab2 = st.tabs(["🖼 画像で検索", "🔤 名前で検索"])

# -------------------------------
# 🖼 画像検索タブ
# -------------------------------
with tab1:
    uploaded = st.file_uploader("トレカ写真をアップロード", type=["jpg", "png", "jpeg"])
    if uploaded:
        query_img = Image.open(uploaded)
        processed = preprocess_card(query_img)
        st.image(processed, caption="前処理後", width=300)
        img_tensor = preprocess(processed).unsqueeze(0).to(device)
        with torch.no_grad():
            q_feat = model.encode_image(img_tensor)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        D, I = index.search(q_feat.cpu().numpy().astype("float32"), 5)

        st.subheader("🔍 類似カード候補")
        for rank, idx in enumerate(I[0]):
            filename = os.path.basename(image_paths[idx])
            meta = info_df[info_df["filename"] == filename]
            name = meta["name"].values[0] if not meta.empty else filename
            series = meta["series"].values[0] if not meta.empty else "-"
            number = meta["number"].values[0] if not meta.empty else "-"
            st.image(
                image_paths[idx],
                caption=f"候補 {rank+1}: {name}\n📀 {series}｜🆔 {number}\n類似度 {D[0][rank]:.3f}",
                width=300,
            )

# -------------------------------
# 🔤 名前検索タブ
# -------------------------------
with tab2:
    query = st.text_input("カード名・シリーズ名・番号で検索", "")
    if query:
        result = info_df[
            info_df.apply(lambda x: query.lower() in str(x).lower(), axis=1)
        ]
        if len(result) == 0:
            st.warning("該当カードが見つかりません。")
        else:
            st.success(f"{len(result)}件ヒットしました。")
            for _, row in result.iterrows():
                image_path = os.path.join(image_folder, row["filename"])
                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=f"{row['name']}\n📀 {row['series']}｜🆔 {row['number']}",
                        width=300,
                    )

import streamlit as st
st.write(" App started")
import torch
from PIL import Image
from torchvision import transforms
from model import EmbeddingNet

st.set_page_config(
    page_title="Image Similarity Search",
    layout="wide"
)

device = "cpu"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@st.cache_resource
def load_model():
    model = EmbeddingNet().to(device)
    model.load_state_dict(
        torch.load("best_embedding_modelx.pth", map_location=device)
    )
    model.eval()
    return model

@st.cache_resource
def load_db():
    embeddings = torch.load("db/embeddings.pt")
    with open("db/paths.txt") as f:
        paths = f.read().splitlines()
    return embeddings, paths

model = load_model()
db_embeddings, db_paths = load_db()

st.title("🔍 Image Similarity Search")
st.write("Upload an image to find similar images")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Query Image", width=250)

    query_tensor = transform(query_img).unsqueeze(0)

    with torch.no_grad():
        query_emb = model(query_tensor)

    similarities = torch.matmul(db_embeddings, query_emb.T).squeeze()
    top5 = torch.topk(similarities, k=5).indices

    st.subheader("Top-5 Similar Images")
    cols = st.columns(5)

    for i, idx in enumerate(top5):
        cols[i].image(db_paths[idx])


import os
import torch
from PIL import Image
from torchvision import transforms
from model import EmbeddingNet

device = "cpu"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

model = EmbeddingNet().to(device)
model.load_state_dict(
    torch.load("best_embedding_modelx.pth", map_location=device)
)
model.eval()

embeddings = []
paths = []

with torch.no_grad():
    for cls in os.listdir("animals5_zip"):
        cls_path = os.path.join("animals5_zip", cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)

            emb = model(img)
            embeddings.append(emb)
            paths.append(img_path)

os.makedirs("db", exist_ok=True)
torch.save(torch.cat(embeddings), "db/embeddings.pt")

with open("db/paths.txt", "w") as f:
    f.write("\n".join(paths))

print("Vector database created successfully")

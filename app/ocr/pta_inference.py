from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "pta_rating_model.pth"
NUM_CLASSES = 5
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_STRONG = 0.65
CONF_AMBIGUOUS = 0.45
MARGIN_AMBIGUOUS = 0.15
# --------------------------------------

_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None


def _load_model():
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m
    return _model


def infer_rating_rows(row_image_paths: List[str]) -> List[Dict]:
    """
    Runs inference on multiple row images.
    """
    model = _load_model()
    results = []

    for path in row_image_paths:
        img = Image.open(path).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        second_prob = float(np.sort(probs)[-2])
        margin = top_prob - second_prob

        if top_idx == 0 and top_prob >= 0.7:
            results.append({
                "value": None,
                "confidence": top_prob,
                "status": "empty_or_noise"
            })
        elif top_idx != 0 and top_prob >= CONF_STRONG:
            results.append({
                "value": top_idx,
                "confidence": top_prob,
                "status": "ok"
            })
        elif top_idx != 0 and top_prob >= CONF_AMBIGUOUS and margin < MARGIN_AMBIGUOUS:
            results.append({
                "value": top_idx,
                "confidence": top_prob,
                "status": "ambiguous"
            })
        else:
            results.append({
                "value": None,
                "confidence": top_prob,
                "status": "empty_or_noise"
            })

    return results

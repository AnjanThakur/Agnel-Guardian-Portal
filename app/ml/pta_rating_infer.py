# app/ml/pta_rating_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


@dataclass
class InferConfig:
    # If env var is present, it wins.
    model_path: str = "pta_rating_model.pth"
    num_classes: int = 5
    image_height: int = 96
    image_width: int = 384

    # thresholds
    conf_strong: float = 0.65
    conf_ambiguous: float = 0.45
    margin_ambiguous: float = 0.15

    # class semantics
    empty_class: int = 0


class PTARatingModel:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: nn.Module | None = None

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.image_height, cfg.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _resolve_model_path(self) -> str:
        env_path = os.getenv("PTA_RATING_MODEL_PATH")
        if env_path:
            return env_path

        # default: same folder as this file (recommended deployment)
        here = Path(__file__).resolve().parent
        candidate = here / self.cfg.model_path
        if candidate.exists():
            return str(candidate)

        # fallback: cwd relative
        return self.cfg.model_path

    def load(self) -> nn.Module:
        if self.model is not None:
            return self.model

        path = self._resolve_model_path()
        if not Path(path).exists():
            raise FileNotFoundError(
                f"PTA rating model not found at '{path}'. "
                f"Place it at app/ml/pta_rating_model.pth or set PTA_RATING_MODEL_PATH."
            )

        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, self.cfg.num_classes)

        state = torch.load(path, map_location=self.device)
        m.load_state_dict(state)

        m.to(self.device)
        m.eval()
        self.model = m
        return m

    def predict_rows(self, row_images_bgr_or_gray: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        row_images: list of numpy images (GRAY or BGR).
        Returns list:
          {"value": int|None, "confidence": float, "status": "...", "probs": [...]}
        """
        self.load()
        assert self.model is not None

        out: List[Dict[str, Any]] = []

        for img in row_images_bgr_or_gray:
            if img is None or getattr(img, "size", 0) == 0:
                out.append({"value": None, "confidence": 0.0, "status": "empty_or_noise", "probs": []})
                continue

            # ensure RGB for model
            if img.ndim == 2:
                rgb = np.stack([img, img, img], axis=-1)
            else:
                # BGR -> RGB
                rgb = img[:, :, ::-1].copy()

            x = self.preprocess(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            second_prob = float(np.sort(probs)[-2])
            margin = top_prob - second_prob

            cfg = self.cfg

            if top_idx == cfg.empty_class and top_prob >= 0.70:
                out.append({
                    "value": None,
                    "confidence": top_prob,
                    "status": "empty_or_noise",
                    "probs": probs.tolist(),
                })
            elif top_idx != cfg.empty_class and top_prob >= cfg.conf_strong:
                out.append({
                    "value": top_idx,
                    "confidence": top_prob,
                    "status": "ok",
                    "probs": probs.tolist(),
                })
            elif top_idx != cfg.empty_class and top_prob >= cfg.conf_ambiguous and margin < cfg.margin_ambiguous:
                out.append({
                    "value": top_idx,
                    "confidence": top_prob,
                    "status": "ambiguous",
                    "probs": probs.tolist(),
                })
            else:
                out.append({
                    "value": None,
                    "confidence": top_prob,
                    "status": "empty_or_noise",
                    "probs": probs.tolist(),
                })

        return out


# ---------------- Singleton helper ----------------

_MODEL_SINGLETON: PTARatingModel | None = None


def infer_rating_rows(row_images: List[np.ndarray]) -> List[Dict[str, Any]]:
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = PTARatingModel(InferConfig())
    return _MODEL_SINGLETON.predict_rows(row_images)

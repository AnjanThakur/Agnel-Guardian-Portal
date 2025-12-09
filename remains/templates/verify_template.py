# verify_template.py
# --------------------------------------------
# Visual debug helper for PTA form template.
# Draws all YAML bounding boxes over the uploaded form.
# Run once to confirm coordinate alignment.
# --------------------------------------------

import cv2, yaml, base64, io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

TEMPLATE_PATH = "templates\pta_parent_feedback.yaml"
FORM_IMAGE_PATH = "templates\PTI Parent Feedback Form_page-0001.jpg"  # your uploaded form image

def load_template(path=TEMPLATE_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def draw_box(img, box, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def visualize_template(form_path=FORM_IMAGE_PATH, template_path=TEMPLATE_PATH, save_path="overlay_debug.png"):
    img = cv2.imread(form_path)
    template = load_template(template_path)
    if img is None:
        print("Error: Could not load form image.")
        return

    for field in template["fields"]:
        ftype = field.get("type", "text")
        color = (0, 255, 0) if ftype in ("text", "paragraph") else (0, 0, 255)

        if ftype == "rating":
            draw_box(img, field["label_box"], color=(255, 165, 0))  # label box in orange
            for rb in field["rating_boxes"]:
                draw_box(img, rb, color=(0, 0, 255))
        else:
            draw_box(img, field["box"], color=color)

    cv2.imwrite(save_path, img)
    print(f"Overlay saved to {save_path}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("PTA Template Overlay")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_template()

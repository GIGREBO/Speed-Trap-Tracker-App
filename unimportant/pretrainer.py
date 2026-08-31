from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
seg_model.eval()

TARGET_CLASSES = [8, 10, 13, 14, 15]  # vegetation, sky, car, truck, bus

def mask_image(image_array):
    image = Image.fromarray(image_array)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = seg_model(**inputs)
    upsampled_logits = F.interpolate(outputs.logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_mask = upsampled_logits.argmax(dim=1)[0].numpy()
    mask_to_remove = np.isin(pred_mask, TARGET_CLASSES)
    masked = image_array.copy()
    masked[mask_to_remove] = [128, 128, 128]
    return masked
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.nn as nn

class ResNetWrapper(nn.Module):
    def __init__(self, model_name="microsoft/resnet-50"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

    def preprocess(self, pil_images):
        return self.processor(images=pil_images, return_tensors="pt")["pixel_values"]

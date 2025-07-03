import torch
from collections import defaultdict

def get_class_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for label, pred in zip(labels, preds):
                total += 1
                per_class_total[label.item()] = per_class_total.get(label.item(), 0) + 1
                if label == pred:
                    correct += 1
                    per_class_correct[label.item()] = per_class_correct.get(label.item(), 0) + 1
    return {cls: per_class_correct.get(cls, 0) / per_class_total[cls] for cls in per_class_total}

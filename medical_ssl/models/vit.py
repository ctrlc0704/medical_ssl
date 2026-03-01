import timm
import torch.nn as nn

def get_vit(model_name="vit_base_patch16_224", pretrained=False):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0  # remove classifier
    )
    return model
    
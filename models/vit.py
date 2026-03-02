import timm

def get_vit(model_name="vit_base_patch16_224", pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model
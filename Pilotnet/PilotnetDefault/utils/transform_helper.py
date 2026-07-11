from torchvision import transforms

def createTransform(augmentations=None):
    transform_list = []

    transform_list.append(transforms.ToTensor())

    # Normalizar como PilotNet: valores entre [0,1]
    transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transform_list)

# Efficient Net infrastructure import
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp

# 1. In order to benefit of the training-predicting workflow as provided
# wrapping the Model's class initialization in a class is required
def get_model():
    return smp.Unet(
        encoder_name="efficientnet-b3",        # Use your chosen backbone
        encoder_weights="imagenet",            # Start with pre-trained knowledge
        in_channels=3,                         # RGB input
        classes=19                            # 19 Cityscapes evaluation classes
    )



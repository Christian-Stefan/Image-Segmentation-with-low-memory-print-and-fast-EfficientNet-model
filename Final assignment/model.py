### --- Start Efficient Net infrastructure imports --- ###
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp
### --- End Efficient Net infrastructure imports --- ###


# 1. In order to benefit of the train.py as provided
# wrapping the Model's class initialization in a class is required;
def get_model():
    return smp.Unet(
        encoder_name="efficientnet-b3",        # Use an eff.net-b3 as encoder
        encoder_weights="imagenet",            # Start with pre-trained on imagenet encoder weights
        in_channels=3,                         
        classes=19                            # 19 Cityscapes evaluation classes
    )


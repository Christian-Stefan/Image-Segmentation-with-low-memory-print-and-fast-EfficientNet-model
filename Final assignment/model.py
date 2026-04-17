# Efficient Net infrastructure import
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp


    ### --- Initial model --- ###
    # Define the model
    # model = model_pretrained(
    #     in_channels=3,  # RGB images
    #     n_classes=19,  # 19 classes in the Cityscapes dataset
    # ).to(device)

    ### --- Second model (the basis) --- ###
    # # model = EfficientNet.from_name('efficientnet-b7') # Loading a not-pretrained model
    # model_pretrained = EfficientNet.from_pretrained('efficientnet-b7', num_classes=19, in_channels=3) # Pretrained

    # backbone_params:list = []
    # forebone_params:list = []

    # for _name, _param in model_pretrained.named_parameters():
    #     if '_fc' in _name:
    #         # # Debug statement 1 
    #         # print("Name {}".format(_name, _param))
    #         forebone_params.append(_param)
    #     else:
    #         backbone_params.append(_param)



# 1. In order to benefit of the training-predicting workflow as provided
# wrapping the Model's class initialization in a class is required
def get_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",        # Use your chosen backbone
        encoder_weights="imagenet",            # Start with pre-trained knowledge
        in_channels=3,                         # RGB input
        classes=19                            # 19 Cityscapes evaluation classes
    )



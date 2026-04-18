import timm
import warnings
warnings.filterwarnings('ignore')

def create_model(num_classes=3):
    model = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in22k_in1k', 
        pretrained=True, 
        num_classes=num_classes
    )
    return model
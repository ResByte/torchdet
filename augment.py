import albumentations as albu
from albumentations import BboxParams
from albumentations.pytorch import ToTensor

def pre_transforms(image_size=512):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=2)
    ]
    
    return result

def hard_transforms():
    result = [
        # random flip 
        albu.Flip(),
        # add random brightness and contrast, 70% prob
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.7
        ),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        # Randomly changes the hue, saturation, and color value of the input image 
        albu.HueSaturationValue(p=0.3),
        # apply compression to lower the image quality 
        albu.JpegCompression(quality_lower=80),
        # add cutout with 0.5 prob on the image 
        albu.Cutout(num_holes=15, max_h_size=20, max_w_size=20, fill_value=5, p=0.5),
        # randomly select of these operations
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
    ]
    
    return result

def post_transforms():
    return [
#         albu.Normalize(), 
        ToTensor()
    ]

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ],bbox_params=BboxParams(format='coco', label_fields=['category_id']))
    return result

def get_transforms(image_size=600):
    """Composes various transformation for train and test settings"""
    
    train_transforms = compose([
                        pre_transforms(image_size), 
                        hard_transforms(), 
                       post_transforms()
    ])
    
    val_transforms = compose([
        pre_transforms(image_size), 
        post_transforms()
    ])
    
    return train_transforms, val_transforms
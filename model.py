import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# from : https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L291 
def create_model(num_classes=2):
    """Creates default resnet50 model with weights pretrained on MSCOCO dataset"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # 2 class: (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def create_model_with_backbone(arch='resnet101', pretrained=True, num_classes=2):
    """creates model with backbone of specified arch and weights pretrained on imagenet"""
    backbone = resnet_fpn_backbone(arch, pretrained=pretrained)

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

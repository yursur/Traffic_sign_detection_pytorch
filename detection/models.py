from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn, faster_rcnn

def create_model(num_classes: int):
    """
    Returns pre-trained Faster RCNN detection model with 'num_classes' classes
    """
    # load Faster RCNN pre-trained model
    detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # get the number of input features
    in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    detection_model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return detection_model



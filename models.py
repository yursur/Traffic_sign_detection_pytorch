from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn

def create_model(num_classes, six_class_detection=False):
    """load pre-trained models"""
    if six_class_detection:
        # load Faster RCNN pre-trained models
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model



import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm
from pprint import pprint
# import matplotlib.pyplot as plt
from models import create_model
from config import DEVICE, NUM_CLASSES, SAVE_MODEL_PATH
from detection.datasets import test_loader

CLASSES = ['background', 'blue_border', 'blue_rect', 'danger', 'main_road', 'mandatory', 'prohibitory']

def test(test_data_loader, model):
    """
    Test function for the DETECTION model.
    Takes every image in 'test_data_loader' and feet it to the model to get predictions.
    This predictions compares with 'ground_truth' targets and the mAP metric calculates.
    Returns mAP.
    """
    print("TESTING")
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    ## initialize tqdm progress bar
    prog_bar = tqdm(test_data_loader, total=len(test_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        ## Get predictions for every image in the batch
        with torch.no_grad():
            output = model(images)
        # print(f'\nOUTPUT:\n{output}\n')
        # print(f"TARGETS:\n{targets}\n")
        # plt.imshow(images[0].permute(1, 2, 0))
        # plt.show()
        ## Calculate mAP for detections on the images of the batch and update 'metric'
        if targets[0]['iscrowd'][0] != 1:
            metric.update(output, targets)
            pprint(metric.compute())
        else:
            ## Pass samples with 'iscrowd' == 1 during evaluation
            # print("\n----------ISCROWD = 1, sample was passed.----------\n")
            continue

    print(f"\nTESTING WAS FINISHED WITH THE RESULTS:\n {metric}")
    return metric


if __name__ == '__main__':
    ## load the model with the trained weights
    model = create_model(num_classes=NUM_CLASSES, six_class_detection=True).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH + '/detection_model14.pth', map_location=DEVICE))
    model.eval()
    tested_model = test(test_loader, model)


import torch
from tqdm.auto import tqdm
import numpy as np
import cv2
from mapcalc import calculate_map, calculate_map_range
from models import create_model
from config import DEVICE, NUM_CLASSES, SAVE_MODEL_ROOT, TEST_IMAGE_ROOT, DETECTION_THRESHOLD

CLASSES = ['background', 'blue_border', 'blue_rect', 'danger', 'main_road', 'mandatory', 'prohibitory']

def test(test_data_loader, model):
    print("Testing")
    for batch_idx, (images, targets) in enumerate(test_data_loader):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

# load the models and the trained weights
model = create_model(num_classes=NUM_CLASSES, six_class_detection=True).to(DEVICE)
model.load_state_dict(torch.load(SAVE_MODEL_ROOT+'/detection_model14.pth', map_location=DEVICE))
model.eval()

for i in range(len(test_images)):

    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i+1] for i in outputs[0]['labels'].cpu().numpy()]


print('TEST PREDICTIONS COMPLETE')

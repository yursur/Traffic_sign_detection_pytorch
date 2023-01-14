import torch
import glob
import numpy as np
import cv2
from models import create_model
from config_detection import DEVICE, NUM_CLASSES, SAVE_MODEL_PATH, INFERENCE_IMAGE_PATH, CONFIDENCE_THRESHOLD, SAVE_IMAGE_PATH

CLASSES = ['background', 'blue_border', 'blue_rect', 'danger', 'main_road', 'mandatory', 'prohibitory']

inference_images = glob.glob(f"{INFERENCE_IMAGE_PATH}/*")
print(f"Inference instances: {len(inference_images)}")

# load the models and the trained weights
model = create_model(num_classes=NUM_CLASSES, six_class_detection=True).to(DEVICE)
model.load_state_dict(torch.load(SAVE_MODEL_PATH + '/detection_model14.pth', map_location=DEVICE))
model.eval()

for i in range(len(inference_images)):
    # get the image file name
    image_name = inference_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(inference_images[i])
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
        print(outputs)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        pred_boxes = outputs[0]['boxes'].data.numpy()
        draw_boxes = pred_boxes.copy()
        # get all the predicited class names and scores
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_scores = outputs[0]['scores'].cpu().numpy()

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            # filter out boxes according to `CONFIDENCE_THRESHOLD`
            if pred_scores[j] > CONFIDENCE_THRESHOLD:
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j],
                            (int(box[0]) - 5, int(box[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),
                            1, lineType=cv2.LINE_AA)
                cv2.putText(orig_image, 'conf: ' + str(round(pred_scores[j],3)),
                            (int(box[0]) - 5, int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),
                            1, lineType=cv2.LINE_AA)
            else: pass
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)
        cv2.imwrite(f"{SAVE_IMAGE_PATH}/{image_name}.jpg", orig_image, )
    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()


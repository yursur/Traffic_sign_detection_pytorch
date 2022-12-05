# Traffic_sign_detection_pytorch
Model for 198 classes (traffic signs) detection+classification with pytorch using Faster-RCNN model

Dataset RTSD for detection, which have 9065 images in train sample and 3022 images in valid sample.

The main idea is to group every signs in 6 classes, detect any sign on image and classify it for 6 classes, and then classify the concrete kind of signs inside each of 6 classes for every detected sign on image.

The project is currently under development



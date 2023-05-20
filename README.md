# Facial Key-points Detection

This project is a PyTorch implementation of a guided project I completed on Coursera titled ***Emotion AI: Facial Key-points Detection***. The original project uses TensorFlow, but I have opted to use PyTorch in this version.

## Project Overview

The goal of this project is to predict facial key points in grayscale images. Each image has 15 key points that the model is trained to predict. The dataset includes 2,140 images.

The model I implemented for this task is a simplified version of the ResNet architecture. You can find the code for the model in `scripts/model.py`.

## Data

The dataset used in this project consists of grayscale images and their corresponding facial key points. Each image has 15 key points, each represented by an x and y coordinate. Here are the data columns:

- `left_eye_center_x`, `left_eye_center_y`
- `right_eye_center_x`, `right_eye_center_y`
- `left_eye_inner_corner_x`, `left_eye_inner_corner_y`
- `left_eye_outer_corner_x`, `left_eye_outer_corner_y`
- `right_eye_inner_corner_x`, `right_eye_inner_corner_y`
- `right_eye_outer_corner_x`, `right_eye_outer_corner_y`
- `left_eyebrow_inner_end_x`, `left_eyebrow_inner_end_y`
- `left_eyebrow_outer_end_x`, `left_eyebrow_outer_end_y`
- `right_eyebrow_inner_end_x`, `right_eyebrow_inner_end_y`
- `right_eyebrow_outer_end_x`, `right_eyebrow_outer_end_y`
- `nose_tip_x`, `nose_tip_y`
- `mouth_left_corner_x`, `mouth_left_corner_y`
- `mouth_right_corner_x`, `mouth_right_corner_y`
- `mouth_center_top_lip_x`, `mouth_center_top_lip_y`
- `mouth_center_bottom_lip_x`, `mouth_center_bottom_lip_y`
- `Image`


Here is an example image from the dataset:

![Example Data](data\example-data.png)


## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
# Facial Key-points Detection
 
This project is a PyTorch implementation of a guided project I completed on Coursera titled ***Emotion AI: Facial Key-points Detection***. The original project uses TensorFlow, but I have opted to use PyTorch in this version.

## Project Overview

The goal of this project is to predict facial key points in grayscale images. Each image has 15 key points that the model is trained to predict. The dataset includes 2,140 images.

The model I implemented for this task is a simplified version of the ResNet architecture. You can find the code for the model in **scripts/model.py**.

## Data

The dataset used in this project consists of grayscale images and their corresponding facial key points. Each image has 15 key points, each represented by an x and y coordinate. The data looks like this:

 Data columns (total 31 columns):
 #   Column                       Dtype  
---  ------                       -----  
 0   left_eye_center_x            float64
 1   left_eye_center_y            float64
 2   right_eye_center_x           float64
 3   right_eye_center_y           float64
 4   left_eye_inner_corner_x      float64
 5   left_eye_inner_corner_y      float64
 6   left_eye_outer_corner_x      float64
 7   left_eye_outer_corner_y      float64
 8   right_eye_inner_corner_x     float64
 9   right_eye_inner_corner_y     float64
 10  right_eye_outer_corner_x     float64
 11  right_eye_outer_corner_y     float64
 12  left_eyebrow_inner_end_x     float64
 13  left_eyebrow_inner_end_y     float64
 14  left_eyebrow_outer_end_x     float64
 15  left_eyebrow_outer_end_y     float64
 16  right_eyebrow_inner_end_x    float64
 17  right_eyebrow_inner_end_y    float64
 18  right_eyebrow_outer_end_x    float64
 19  right_eyebrow_outer_end_y    float64
 20  nose_tip_x                   float64
 21  nose_tip_y                   float64
 22  mouth_left_corner_x          float64
 23  mouth_left_corner_y          float64
 24  mouth_right_corner_x         float64
 25  mouth_right_corner_y         float64
 26  mouth_center_top_lip_x       float64
 27  mouth_center_top_lip_y       float64
 28  mouth_center_bottom_lip_x    float64
 29  mouth_center_bottom_lip_y    float64
 30  Image                        object 

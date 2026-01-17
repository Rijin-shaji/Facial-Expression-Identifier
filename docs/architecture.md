# Model Architecture

The model is based on **MobileNetV2**, a lightweight CNN optimized for performance and efficiency.

## Architecture Overview
- Pretrained MobileNetV2 backbone (ImageNet)
- Global Average Pooling
- Fully connected Dense layer (ReLU)
- Dropout for regularization
- Softmax output layer

## Advantages
- Faster training
- Lower computational cost
- Suitable for real-time applications

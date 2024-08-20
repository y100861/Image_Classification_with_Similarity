# 01. Image Data
   - Celebrity Faces Dataset
   - 17 people 100 images of each person (images of Scarlett Johansson are 200)
   - All 1800 images (train: 1260, validation: 360, test: 180)
   - Data augmentation (Resize(112 $\times$ 112), Horizontal Flip, Color Jitter, Normalize)
   - Train batch size: 64
   - Validation batch size: 16
     

# 02. Model (SimpleCNN)
   - 4 Layers (Convolution - BatchNorm - ReLU - MaxPool - AvgPool)
   - Convolution
      - initial in_channels: 3
      - final out_channels: 128
      - kernel size: 3 $\times$ 3
      - stride: 2
      - padding: 2
   - MaxPool
      - kernel size: 2 $\times$ 2
      - stride: 2
      - padding: 0


# 03. Loss Function, Optimizer, Scheduler
   - Loss Function: CrossEntropyLoss
   - Optimizer: Adam
      - learning rate: 0.001
      - betas: 0.9, 0.999
      - weight decay: 0.0004
   - Scheduler: StepLR (multiply learning rate by gamma each step size)
      - step size: 5
      - gamma: 0.8

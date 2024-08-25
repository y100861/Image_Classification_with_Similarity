# 01. Image Data
   - Celebrity Faces Dataset
   - 17 people 100 images of each person (images of Scarlett Johansson are 200)
   - All 1800 images (train: 1260, validation: 360, test: 180)
   - Data augmentation (Resize(112 $\times$ 112), Horizontal Flip, Color Jitter, Normalize)
   - Train batch size: 64
   - Validation batch size: 16
     

# 02. Model (SimpleCNN)
   - 5 Layers (Convolution - BatchNorm - ReLU - MaxPool - AvgPool)
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
   - Epochs: 100
   - Loss Function: CrossEntropyLoss
   - Optimizer: Adam
      - learning rate: 0.01
      - betas: 0.9, 0.999
   - Scheduler: StepLR (multiply learning rate by gamma each step size)
      - step size: 10
      - gamma: 0.8


# 04. Result
   - Train Loss decreased rapidly, and Train Accuracy increased rapidly too.
   - Initially, Validation Loss decreased rapidly, and Validation Accuracy increased rapidly.
      - But at some point, performance of model with validation data was stagnant. (best accuracy: 0.54~)
   - It was overfitting.


# 05. Measuring Similarity
   - Made gallery and query to measure similarity.
   - Used "IndexFlatL2" of Faiss.
   - Showed quite a high value of distances.
   - Accuracy score with similarity 0.47~
   - It showed overfitting too.


# 06. Another Model (InceptionResnetV1)
   - Wanted better perfomance.
   - So searched about better model about image classification.
   - And I brought the "InceptionResnetV1" code from the Facenet-pytorch.
   - Hyperparameter
      - Epochs: 100
      - Loss Function: CrossEntropyLoss
      - Optimizer: Adam
        - learning rate: 0.001
        - betas: 0.9, 0.999
      - Scheduler: StepLR (multiply learning rate by gamma each step size)
        - step size: 10
        - gamma: 0.95


# 07. Result
   - Train Accuracy and Validation Accuracy increased rapidly.
   - At some point Validation Accuracy was stagnant. (best accuracy: 0.63~)
   - But, best Train Accuracy resulted 0.98~
   - It showed some overfitting but better than SimpleCNN

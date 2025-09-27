# Mediskin: Skin Disease Detection Web App
Mediskin is a web-based application designed to assist in the early detection of skin diseases using image classification.  
Users can upload a photo of a skin lesion, and the system processes the image through a deep learning model to predict the possible disease category.  

Before the model was integrated into the web application, an exploratory phase was carried out by experimenting with several pre-trained models, including **MobileNetV2, Xception, and DenseNet121**.  During this stage, hyperparameter tuning (such as learning rate adjustment) and K-Fold Cross Validation were performed to evaluate model performance and determine the most optimal architecture

## Dataset
The dataset used in this project was collected from public sources: Monkeypox Skin Images Dataset (MSID) v6, which is available on Mendeley Data https://data.mendeley.com/datasets/r9bfpnvyxr/6

It contains images categorized into four classes: Monkeypox, Chickenpox, Measles, and Healthy (normal skin)

## Model Architecture
The model uses transfer learning with pre-trained CNNs (MobileNetV2, Xception, DenseNet121) combined with custom layers for classification. The added layers include:
- Global Average Pooling
- Dense (Fully Connected) layers with ReLU
- Dropout
- Softmax output
<img width="1140" height="347" alt="image" src="https://github.com/user-attachments/assets/86e64ff4-9b81-421a-9c85-8366833ea975" />


## Final Model Result
The best performing model was **MobileNetV2 with learning rate 0.0001**
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/049a72af-103e-4e52-ade8-c6194db98994" />


## Web Application
To make the model accessible and user-friendly, it was integrated into a web-based application called Mediskin. The web app allows users to:
- Upload an image of a skin lesion for classification
- Receive predictions along with the probability of each class
- View recommendation related to the predicted disease for educational purposes
- Log in or Sign up to store and track personal prediction history

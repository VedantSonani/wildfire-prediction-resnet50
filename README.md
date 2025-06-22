🔥 Wildfire Prediction Using ResNet50
This project predicts wildfire occurrence using satellite imagery and a convolutional neural network based on the pretrained ResNet50 model. Built in Google Colab, the pipeline includes image preprocessing, model fine-tuning, and performance visualization. It demonstrates the power of transfer learning for binary image classification in environmental monitoring.

📁 Dataset
Source: Kaggle - Wildfire Prediction Dataset

Classes: wildfire, no_wildfire

Input format: RGB images (resized to 224×224)

🧠 Model Architecture
Base model: ResNet50 (weights='imagenet', include_top=False)

Added layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense(128, ReLU) → Dropout(0.3)

Output: Dense(1, Sigmoid)

Model is compiled with binary_crossentropy and Adam optimizer.

🛠️ Libraries Used
TensorFlow / Keras

PIL

matplotlib

Kaggle API (for dataset download)

Google Colab environment

📦 Setup (Google Colab)
Upload your kaggle.json to authenticate Kaggle API:

python
Copy
Edit
from google.colab import files
files.upload()
Move and secure credentials:

bash
Copy
Edit
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Download and unzip dataset:

bash
Copy
Edit
!kaggle datasets download -d abdelghaniaaba/wildfire-prediction-dataset -p /content
!unzip -q /content/wildfire-prediction-dataset.zip -d /content/wildfire_dataset
🚀 Training
python
Copy
Edit
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[EarlyStopping(...), ReduceLROnPlateau(...)]
)
Training and validation data is handled via ImageDataGenerator with preprocessing using ResNet50-compatible format.

📊 Visualization
Accuracy and Loss curves plotted using matplotlib

Evaluation metrics shown after training

📈 Results
Achieves competitive binary classification accuracy on validation set

Learning rate adjustments and early stopping used to avoid overfitting

Face Mask Detection
This project is designed to build a face mask detection system using a Convolutional Neural Network (CNN) model in Keras. The model is trained to classify images as either wearing a mask or not wearing a mask.

Table of Contents
Installation
Usage
Project Structure
Model Training
Dependencies
Acknowledgements
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/facemask-detection.git
cd facemask-detection
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Train the model:

python
Copy code
python train_model.py
Evaluate the model:

python
Copy code
python evaluate_model.py
Make predictions:

python
Copy code
python predict.py --image path/to/image.jpg
Project Structure
bash
Copy code
facemask-detection/
├── data/
│   ├── train/
│   ├── test/
├── models/
│   ├── cnn.keras
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
├── requirements.txt
└── README.md
Model Training
To train the model, run the train_model.py script. The model architecture is defined as follows:

python
Copy code
import tensorflow as tf
from tensorflow.keras import layers, models

img_shape = (50, 50, 3)
model = models.Sequential()
model.add(layers.InputLayer(input_shape=img_shape))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))

model.add(layers.Dense(2, activation='softmax'))

adam = tf.keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Assuming trainx and trainy are already defined
history = model.fit(x=trainx, y=trainy, batch_size=100, epochs=50, validation_split=0.2, verbose=2, shuffle=True)

# Save the model in the new format
model.save('cnn.keras')
Dependencies
The project requires the following Python libraries:

tensorflow
keras
numpy
matplotlib
opencv-python
These dependencies are listed in the requirements.txt file and can be installed using:

bash
Copy code
pip install -r requirements.txt
Acknowledgements
This project uses the TensorFlow and Keras libraries for building and training the neural network model. The dataset used for training and testing the model should be sourced from appropriate datasets available for face mask detection.


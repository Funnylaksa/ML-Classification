# Your model will be tested as following
import tensorflow as tf

# Model reconstruction from JSON file
with open('LeeChangHsien.json', 'r') as json_file:
    json_savedModel= json_file.read()

test_model = tf.keras.models.model_from_json(json_savedModel)
test_model.summary()

test_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                    metrics=['acc'])

# Load weights into the new model
test_model.load_weights('LeeChangHsien.h5')


import pandas as pd
import numpy as np
import cv2
data_path = 'test.csv'
image_size=(48, 48)

def load_data(data_path):
        data = pd.read_csv(data_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        emotions = data.emotion.values.reshape(-1, 1)
        return faces, emotions
    
faces_test, emotions_test = load_data(data_path); 


## Testing
test_loss, test_acc = test_model.evaluate(faces_test, emotions_test) 
print('Test accuracy:', test_acc)

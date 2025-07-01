from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image, model_type):
    img = Image.open(image).convert('RGB')
    if "VGG-19" in model_type:
        img = img.resize((224, 224))
        img_array = tf.keras.applications.vgg19.preprocess_input(np.array(img))
    elif "Xception" in model_type:
        img = img.resize((299, 299))
        img_array = tf.keras.applications.xception.preprocess_input(np.array(img))
    else:
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
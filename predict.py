import tensorflow as tf
from model import MLPMixer
from argparse import ArgumentParser
import os
import numpy as np


if __name__ == "__main__":
    home_dir = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument("--test-file-path", default='{}/data/test'.format(home_dir), type=str, required=True)
    parser.add_argument("--model-folder", default='{}/model/mlp/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=150, type=int)
    
    args = parser.parse_args()
    print('---------------------Welcome to ProtonX MLP Mixer-------------------')
    print('Github: bangoc123')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Predict using MLP Mixer for image path: {}'.format(args.test_file_path))
    print('===========================')


    # Loading Model
    mlpmixer = tf.keras.models.load_model(args.model_folder)

    # Load test images from folder
    image = tf.keras.preprocessing.image.load_img(args.test_file_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    x = tf.image.resize(
        input_arr, [args.image_size, args.image_size]
    )

    predictions = mlpmixer.predict(x)   
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(predictions))
    print('This image belongs to class: {}'.format(np.argmax(predictions), axis=1))


import tensorflow as tf
from tensorflow import keras
import numpy as np

if __name__ == '__main__':
    model = tf.keras.models.load_model('my_model_step1.keras')
    model.summary()

    img_height = 180
    img_width = 180

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # Getting from google apis
    # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    sunflower_path = '592px-Red_sunflower.jpg'

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

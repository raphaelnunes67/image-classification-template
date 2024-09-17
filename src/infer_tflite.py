import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Load Tensorflowlite
    interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
    interpreter.allocate_tensors()

    # Obtain tensors details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_height = 180
    img_width = 180

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # Getting from google apis
    # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    sunflower_path = '592px-Red_sunflower.jpg'

    # Prepare image
    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)

    # Defines entry data for model
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Peform the inference
    interpreter.invoke()

    # Obtain results
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Normalize using softmax
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

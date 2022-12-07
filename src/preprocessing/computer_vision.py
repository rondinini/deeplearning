import tensorflow as tf

def scaling(tensor):
    return tensor / 255

def reshaping(image, pixels):
    return tf.reshape(image, (pixels, pixels, -1))

def resize(image, target_height, target_width):
    return tf.image.resize_with_pad(image, target_height, target_width)

def augmentation(image):       
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0.5, 1)
    image = tf.image.random_hue(image, 0.1)
    return image
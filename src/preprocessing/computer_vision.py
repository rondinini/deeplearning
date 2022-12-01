import tensorflow as tf

def scaling(tensor):
    return tensor / 255

def reshaping(image, pixels):
    return tf.reshape(image, (pixels, pixels, -1))

def resize(image, target_height, target_width):
    return tf.image.resize_with_crop_or_pad(image, target_height, target_width)

def augmentation(image):       
    image = tf.image.random_contrast(image, 0.2, 0.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image
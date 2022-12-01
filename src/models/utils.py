import tensorflow as tf

def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.4,
            verbose=1,
            patience=5, 
            min_lr=0.001
        )
    ]
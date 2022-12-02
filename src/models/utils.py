import tensorflow as tf

def callbacks(tensorboard_logdir):
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
            patience=10, 
            min_lr=0.0005
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_logdir,
            histogram_freq=2,
            write_graph=True,
            write_images=True,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
    ]
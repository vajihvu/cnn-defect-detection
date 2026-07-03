import tensorflow as tf

def get_augmentation_layer():

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomContrast(0.2)
    ], name="data_augmentation")

def get_baseline_model(img_size=224):

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    

    x = get_augmentation_layer()(inputs)
    x = tf.keras.layers.Rescaling(1./255)(x)
    

    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    return model

def get_transfer_model(img_size=224, freeze_backbone=True):

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    

    x = get_augmentation_layer()(inputs)
    

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    if freeze_backbone:
        base_model.trainable = False
        
    x = base_model(x, training=not freeze_backbone)
    

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="efficientnet_cnn")
    return model

def BaselineCNN():
    return get_baseline_model()

def TransferLearningCNN(freeze_backbone=True):
    return get_transfer_model(freeze_backbone=freeze_backbone)

if __name__ == "__main__":

    baseline = BaselineCNN()
    effnet = TransferLearningCNN()
    
    baseline.summary()
    effnet.summary()

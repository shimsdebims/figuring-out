import tensorflow as tf
model = tf.keras.models.load_model("Model/crop_model.h5")
print(model.summary())
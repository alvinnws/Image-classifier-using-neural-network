import tensorflow as tf

model = tf.keras.models.load_model("best")
model.summary()
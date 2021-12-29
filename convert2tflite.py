import tensorflow as tf

TF_PATH = "." 
TFLITE_PATH = "mobilenet_v2.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)

import tensorflow as tf

# Load your .keras model
model = tf.keras.models.load_model('traffic-signs.keras')

model.save("traffic-signs.h5")

model_json = model.to_json()
with open("traffic-signs.json", "w") as json_file:
    json_file.write(model_json)
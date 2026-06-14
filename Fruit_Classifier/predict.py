import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("fruit_classifier.h5")

class_names = ["Apple", "Banana", "Orange"]

img = image.load_img(
    "test_images/banana.jpg",
    target_size=(100,100)
)

plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print(prediction)

print("Apple :", prediction[0][0] * 100)
print("Banana:", prediction[0][1] * 100)
print("Orange:", prediction[0][2] * 100)

index = np.argmax(prediction)

print("Prediction:", class_names[index])
print("Confidence:", np.max(prediction) * 100)
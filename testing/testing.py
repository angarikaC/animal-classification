from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model=load_model("animalsbatao3.h5")
img=image.load_img('elephant.jpeg',target_size=(128,128))
img_array=image.img_to_array(img)/255
img_array=np.expand_dims(img_array,axis=0)

predict=model.predict(img_array)
print(predict)
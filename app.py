import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#resize the image to a 224x224 with the same strategy as in TM2:
  #resizing the image to be at least 224x224 and then cropping from the center

  #turn the image into a numpy array


while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    img = js_to_image(js_reply["img"])
    #size = (224, 224)
    #imag2 = ImagOps.fit(img, size, Imag.ANTIALIAS)

    imag2= cv2.resize(img, (224, 224),
               interpolation = cv2.INTER_LINEAR)


    image_array = np.asarray(imag2)
  # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
    data[0] = normalized_image_array

  # run the inference
    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] >0.6:
       print('Abierto: ')
       client1.publish("IMIA","{'gesto': 'Abierto'}",qos=0, retain=False)
       #sound_file = 'hum_h.wav'
       #display(Audio(sound_file, autoplay=True))
       time.sleep(0.5)
    if prediction[0][1]>0.6:
       print('Cerrado')
       client1.publish("IMIA","{'gesto': 'Cerrado'}",qos=0, retain=False)
       time.sleep(0.5)
    if prediction[0][2]>0.6:
       print('Vacío')
       client1.publish("IMIA","{'gesto': 'Vacío'}",qos=0, retain=False)
       time.sleep(0.5)

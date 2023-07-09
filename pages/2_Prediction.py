import streamlit as st
import os
from matplotlib.cbook import file_requires_unicode
from PIL import Image
import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt
from keras.preprocessing import image
import keras.utils as image
from keras.utils import load_img
import numpy as np

st.header("UPLOAD IMAGE TO PREDICT")
Classes = [":red[UnHealthy Leaf]: Apple___Apple_scab", 
           ":red[UnHealthy Leaf]: Apple___Black_rot", 
           ":red[UnHealthy Leaf]: Apple___Cedar_apple_rust", 
           ":red[Healthy Leaf]: Apple___healthy", 
           ":red[Healthy Leaf]: Blueberry___healthy", 
           ":red[UnHealthy Leaf]: Cherry_(including_sour)___Powdery_mildew", 
           ":red[Healthy Leaf]: Cherry_(including_sour)___healthy",
           ":red[UnHealthy Leaf]: Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
           ":red[UnHealthy Leaf]: Corn_(maize)___Common_rust",  
           ":red[UnHealthy Leaf]: Corn_(maize)___Northern_Leaf_Blight", 
           ":red[Healthy Leaf]: Corn_(maize)___healthy",
           ":red[UnHealthy Leaf]: Grape___Black_rot", 
           ":red[UnHealthy Leaf]: Grape__Esca(Black_Measles)", 
           ":red[UnHealthy Leaf]: Grape__Leaf_blight(Isariopsis_Leaf_Spot)",  
           ":red[Healthy Leaf]: Grape___healthy",
           ":red[UnHealthy Leaf]: Orange__Haunglongbing(Citrus_greening)", 
           ":red[UnHealthy Leaf]: Peach___Bacterial_spot", 
           ":red[Healthy Leaf]: Peach___healthy", 
           ":red[UnHealthy Leaf]: Pepper_bell__Bacterial_spot", 
           ":red[Healthy Leaf]: Pepper_bell__healthy", 
           ":red[UnHealthy Leaf]: Potato___Early_blight",
           ":red[UnHealthy Leaf]: Potato___Late_blight",
           ":red[Healthy Leaf]: Potato___healthy",
           ":red[Healthy Leaf]: Raspberry___healthy", 
           ":red[Healthy Leaf]: Soybean___healthy", 
           ":red[UnHealthy Leaf]: Squash___Powdery_mildew", 
           ":red[UnHealthy Leaf]: Strawberry___Leaf_scorch",
           ":red[Healthy Leaf]: Strawberry___healthy",
           ":red[UnHealthy Leaf]: Tomato___Bacterial_spot",
           ":red[UnHealthy Leaf]: Tomato___Early_blight",
           ":red[UnHealthy Leaf]: Tomato___Late_blight",
           ":red[UnHealthy Leaf]: Tomato___Leaf_Mold",
           ":red[UnHealthy Leaf]: Tomato___Septoria_leaf_spot",
           ":red[UnHealthy Leaf]: Tomato___Spider_mites Two-spotted_spider_mite",
           ":red[UnHealthy Leaf]: Tomato___Target_Spot",
           ":red[UnHealthy Leaf]: Tomato___Tomato_Yellow_Leaf_Curl_Virus" ,
           ":red[UnHealthy Leaf]: Tomato___Tomato_mosaic_virus",
           ":red[Healthy Leaf]: Tomato___healthy",
           ]
image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
def load_image(image_file):
    img = Image.open(image_file)
    return img
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    file_name=list(file_details.values())
    img_name=file_name[0]
    img = load_image(image_file)
    with open(os.path.join("./images/",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("Saved")

clk=st.button("Predict")
model=tf.keras.models.load_model('Plant_final_1.h5') 

if clk:  
    directory="./images/{}".format(img_name)
    #directory=file_name[0]
    #files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
    #for i in range(0,5):
    image_path = directory
    new_img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    #print(prediction)
    probabilty = prediction.flatten()
    #print(probabilty)
    max_prob = probabilty.max()
    index=prediction.argmax(axis=-1)[0]
    #print(index)
    class_name = Classes[index]
    #print(class_name)
    #ploting image with predicted class name        
    plt.figure(figsize = (4,4))
    #ax = plt.subplot(3,3, i + 1)
    plt.imshow(new_img)
    st.image(new_img)
    plt.axis('off')
    plt.title(class_name+" "+ str(max_prob)[0:4]+"%")
    st.write(":blue[Predicted Image]")
    st.write(class_name)
    plt.show()




#files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
    #st.write(directory)
    #st.write(directory)
    # st.write(files[0])
    # st.image(files)

    # directory="./images/"
    # files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))] 
        
    #remove_file=files[0]
    #os.remove(files[0])


    
      
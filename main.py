import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path

from PIL import Image

import random

from streamlit_option_menu import option_menu

import splitfolders

from keras.models import load_model
from keras.utils import load_img,img_to_array



main_dir = Path("Original Images")
monkey_dir ="Original Images/Monkey Pox"
others_dir = "Original Images/others"
monkey_glob = list(main_dir.glob("Monkey Pox/*.jpg"))
others_glob=list(main_dir.glob("Others/*.jpg"))


#check for imbalanced data
data=[]
for i in monkey_glob:
    data.append((i,'Monkeypox'))

for k in others_glob:
    data.append((k,'others'))

print(data)
# construct dataset code
data_histo = pd.DataFrame(data)
data_histo=data_histo.rename({0:'Labels',1:'Count'},axis=1)

data_histo_summ=data_histo['Count'].value_counts()
print(data_histo_summ)
data_histo_table = pd.DataFrame({'Labels':data_histo_summ.index,'Count':data_histo_summ.values})
print(data_histo_table)
# create a list of images and a list of labels

images=[]
labels=[]
height=150
width=150

splitfolders.ratio("Original Images",output='output6',seed=111,ratio=(0.6,0.3,0.1))

train_ds=keras.preprocessing.image_dataset_from_directory(
    './output6/train',
    image_size=(height,width)
)
val_ds=keras.preprocessing.image_dataset_from_directory(
    './output6/val',
    image_size=(height,width)
)
test_ds=keras.preprocessing.image_dataset_from_directory(
    './output6/val',
    image_size=(height,width)
)
class_names=train_ds.class_names

#transform train_ds originally images to augmented images
def augment_data(image,label):
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_brightness(image,max_delta=0.1)
    return image, label

# Apply data augmentation to the batch dataset
augmented_dataset = train_ds.map(augment_data)




with st.sidebar:
    choose = option_menu('App Gallery',['About','Monkeypox images','Non Monkeypox images','Images Augmentation','AI-Predict'],
                         icons=['house','image','image-fill','image-alt','question-diamond-fill'],
                         menu_icon='prescription2',default_index=0,
                         styles={
                             'container':{'padding':"5!important","background-color":"#fafafa"},
                             'icon':{"color":"orange","font-size":"25px"},
                             "nav-link":{"font-size":"16px","text-align":"left","margin":"0px","--hover-color":"#eee"},
                             "nav-link-selected": {"background-color":"#02ab21"},
                         })

if choose=='About':
    st.write("<h2>Monkeypox Skin Lesion Dataset<h2>",unsafe_allow_html=True)
    st.write("The dataset is collected from Kaggle, it includes **102 Monkeypox** images and **126 for others**. This is a binary classification problem to predict Monkeypox Vs Others (Chickenpox, Measles) and we will use Deep Learning with CNN using Tensorflow and Keras to build the model architecture")

elif choose=='Monkeypox images':
    st.write("<div align='center'><h3>Monkeypox Images<h3></div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    available_images=[]
    with col1:
        for i in range(2):
            rand1 = random.randint(0,20)
            if rand1 not in available_images:
                img1=Image.open(monkey_glob[rand1])
                st.image(img1)
                available_images.append(rand1)
    with col2:
        for k in range(2):
            random2=random.randint(20,40)
            if random2 not in available_images:
                img2=Image.open(monkey_glob[random2])
                st.image(img2)
                available_images.append(random2)
    with col3:
        for p in range(2):
            random3=random.randint(40,60)
            if random3 not in available_images:
                img3=Image.open(monkey_glob[random3])
                st.image(img3)
                available_images.append(random3)

elif choose=='Non Monkeypox images':
    avail_images=[]
    st.write("<div align='center'><h3>Non Monkeypox Images<h3></div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        for i in range(2):
            rand4=random.randint(0,20)
            if rand4 not in avail_images:
                img4=Image.open(others_glob[rand4])
                st.image(img4)
                avail_images.append(rand4)
    with col2:
        for k in range(2):
            rand5=random.randint(20,40)
            if rand5 not in avail_images:
                img5=Image.open(others_glob[rand5])
                st.image(img5)
                avail_images.append(rand5)
    with col3:
        for j in range(2):
            rand6=random.randint(40,60)
            if rand6 not in avail_images:
                img6=Image.open(others_glob[rand6])
                st.image(img6)
                avail_images.append(rand6)


# generate and plot augmented images
elif choose=='Images Augmentation':
    st.write("<div align='center'><h3>Data Augmentation<h3></div>",unsafe_allow_html=True)
    st.write("The data is augmented so the model can generalize better on the unseen images, the plan is to avoid overfitting, the data augmentation includes flipping and additional brightness.")
    st.write("")
    st.write("<div align='center'><h4>Click again to get new images<h4></div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        for images,labels in augmented_dataset.take(1):
            rand1=random.randint(0,10)
            img=images[rand1].numpy().astype('uint8')
            img=cv2.resize(img,(300,300))
            st.write ( class_names[labels[rand1]] )
            st.image(img)

    with col2:
        for images,labels in augmented_dataset.take(1):
            rand2=random.randint(10,20)
            img1=images[rand2].numpy().astype('uint8')
            img1=cv2.resize(img1,(300,300))
            st.write(class_names[labels[rand2]])
            st.image(img1)

    with col3:
        for images,labels in augmented_dataset.take(1):
            rand3=random.randint(20,30)
            img2=images[rand3].numpy().astype('uint8')
            img2=cv2.resize(img2,(300,300))
            st.write(class_names[labels[rand3]])
            st.image(img2)

elif choose=='AI-Predict':
    model=load_model('monkey_pox1.h5')
    image_paths1=[('monkeypox1.jpg','Monkeypox')]
    image_path2 =[('Others1.jpg','Chickenpox or Measles')]
    image_path3 =[('monkeypox2.jpg','Monkeypox')]
    image_path4=[('Others2.jpg','Chickenpox or Measles')]

    st.title("Image Classification")
    class_names = ['Monkeypox', 'Others']
    uploaded_file = st.file_uploader("",type=['jpg','jpeg','png'])
    if st.button('Predict'):
        if uploaded_file is not None:
            img=load_img(uploaded_file,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=img/255.0
            pred= model.predict(img)
            arg_max=np.argmax(pred)
            pred_int = pred[arg_max][0]
            if pred>0.5:
                st.write(f"The model is {round(pred_int*100,2)}% confident that the image shows NO signs of Monkeypox")
            else:
                st.write(f"The model is {round((1-pred_int)*100,2)}% confident that the image shows signs of Monkeypox")

    col1,col2,col3,col4=st.columns(4)
    with col1:
        for path, label in image_paths1:
            image = Image.open ( path )
            st.image ( image, caption=label )
    with col2:
        for path,label in image_path2:
            image=Image.open(path)
            st.image(image,caption=label)
    with col3:
        for path,label in image_path3:
            image=Image.open(path)
            st.image(image,caption=label)
    with col4:
        for path,label in image_path4:
            image=Image.open(path)
            st.image(image,caption=label)





































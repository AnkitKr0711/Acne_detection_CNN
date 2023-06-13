# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:07:13 2023

@author: ankro
"""
# Flask API Model deployment
from flask import Flask, render_template, request
from keras.models import load_model
from img_process import rmvbgr

model  = load_model('weights-improvement-19-0.69.h5')


app=Flask(__name__)
app.secret_key = 'your-secret-key'

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded."
    
    image_file = request.files['image']
    
    # Check if the file is actually an image
    if not image_file.filename.endswith(('.jpg', '.jpeg', '.png')):
        return "Invalid image format. Supported formats: JPG, JPEG, PNG" 
    
    img = rmvbgr(image_file.read())
    print('final_image',type(img),img.shape)
    result = model.predict(img)
    
    return render_template('index.html',result=result)


if __name__ =='__main__':
    app.run(debug=True)

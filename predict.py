# pages/api/predict.py

from flask import Flask, request,jsonify
import os, io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

from PIL import Image

app = Flask(__name__)

# Load the Keras model
model_path = os.path.join("models", "model.h5")
model = load_model(model_path)

# Define class labels (ref)
ref = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']# Replace with your class labels

# To access send POST request along with form-data, image : (IMAGE.JPEG)
@app.route('/api/predict', methods=['POST'])
def predictionX():
    try:
        # Get the image file from the request
        file = request.files['image']
        print(file.filename)
        
        # Read image data from the request
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Resize the image to the target size
        img = image.resize((256, 256))

        
        i = img_to_array(img)
        im = preprocess_input(i)
        img = np.expand_dims(im, axis=0)

        # Make predictions
        pred = np.argmax(model.predict(img))

        # Get the predicted class label
        predicted_class = ref[pred]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)



# to run first -> env\scripts\activate -> python predict.py 

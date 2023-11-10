from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

model = load_model('sweets.h5')

@app.route('/')
def home():
    return render_template('home.html')
    
def predict_label(uploaded_image):
    if uploaded_image:
        # Process the image with your model
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))  # Adjust to match the model's input size

        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values

        # Make predictions using the model
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        
        # Define your class labels
        class_labels = ['ariselu', 'basundi', 'boondi', 'chikki', 'doodhpak', 'gavvalu', 'gulab jamun',
                'halwa', 'jalebi', 'kajjikaya', 'kakinada khaja', 'kalakand', 'laddu' , 'mysore pak',
                'poornalu', 'pootharekulu', 'ras malai', 'rasgulla', 'sheer', 'soan papdi']

        # Get the predicted food class
        predicted_food = class_labels[predicted_class_index]

        return predicted_food
    return "No image Uploaded"


@app.route("/submit", methods = ['GET', 'POST'])
def get_food_name():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/"+img.filename
        img.save(img_path)
        p = predict_label(img)
    return render_template("home.html", prediction = p,img_path=img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
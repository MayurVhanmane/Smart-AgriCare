# Importing essential libraries and modules

# from flask import Flask, render_template, request, Markup
import webbrowser
import os
from urllib.parse import urlsplit
from threading import Timer
from markupsafe import Markup, escape
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from translations import PREDICTION_TRANSLATIONS, TRANSLATIONS
import requests
import config
import pickle
import io
import torch
from PIL import Image
from utils.model import ResNet9
from flask import Flask, render_template, request, session, redirect, url_for

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
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
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_scaler = pickle.load(open('./models/crop_scaler.pkl','rb'))
model = pickle.load(open('./models/crop_recommendation_model (1).pkl', 'rb'))

# Loading crop recommendation model

# crop_recommendation_model_path = 'models/RandomForest.pkl'
# crop_recommendation_model = pickle.load(
#     open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    image = Image.open(io.BytesIO(img)).convert('RGB')
    width, height = image.size

    # Match torchvision.transforms.Resize(256): resize the shorter edge and
    # preserve the image aspect ratio.
    if width <= height:
        resized_size = (256, int(256 * height / width))
    else:
        resized_size = (int(256 * width / height), 256)
    image = image.resize(resized_size, resample=Image.BILINEAR)

    # Match ToTensor() for RGB images: convert HWC uint8 pixels to CHW floats
    # in the [0, 1] range.
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1)))
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32)


@app.context_processor
def inject_language():
    language = session.get('language', 'en')

    def translate(text):
        return TRANSLATIONS.get(language, {}).get(text, text)

    crop_names = {
        'rice': 'Rice',
        'maize': 'Maize',
        'chickpea': 'Chickpea',
        'kidneybeans': 'Kidney beans',
        'pigeonpeas': 'Pigeon peas',
        'mothbeans': 'Moth beans',
        'mungbean': 'Mung bean',
        'blackgram': 'Black gram',
        'lentil': 'Lentil',
        'pomegranate': 'Pomegranate',
        'banana': 'Banana',
        'mango': 'Mango',
        'grapes': 'Grapes',
        'watermelon': 'Watermelon',
        'muskmelon': 'Muskmelon',
        'apple': 'Apple',
        'orange': 'Orange',
        'papaya': 'Papaya',
        'coconut': 'Coconut',
        'cotton': 'Cotton',
        'jute': 'Jute',
        'coffee': 'Coffee',
    }

    def translate_crop_name(value):
        english_name = crop_names.get(str(value).lower(), str(value).title())
        return translate(english_name)

    return {
        't': translate,
        'crop_name': translate_crop_name,
        'current_language': language
    }


def localized_disease_result(predicted_label):
    language = session.get('language', 'en')
    if language == 'en':
        return Markup(str(disease_dic[predicted_label]))

    localized = PREDICTION_TRANSLATIONS[language]
    crop_key, disease_key = predicted_label.split('___', 1)
    crop = localized['disease_crops'].get(crop_key, crop_key.replace('_', ' '))
    disease = localized['disease_names'].get(disease_key, disease_key.replace('_', ' '))
    description = (
        localized['healthy_message']
        if disease_key == 'healthy'
        else localized['disease_message']
    )
    return Markup('<strong>{} - {}</strong><br><br>{}').format(
        escape(crop), escape(disease), escape(description)
    )


@app.route('/set-language', methods=['POST'])
def set_language():
    language = request.form.get('language', 'en')
    if language in ('en', 'mr', 'hi'):
        session['language'] = language

    next_page = request.form.get('next', '/')
    try:
        parsed_next = urlsplit(next_page)
        is_local_path = (
            not parsed_next.scheme
            and not parsed_next.netloc
            and parsed_next.path.startswith('/')
            and not parsed_next.path.startswith('//')
            and '\\' not in parsed_next.path
        )
    except ValueError:
        is_local_path = False
    if not is_local_path:
        next_page = '/'
    next_page = {
        url_for('crop_prediction'): url_for('crop_recommend'),
        url_for('fert_recommend'): url_for('fertilizer_recommendation'),
    }.get(next_page, next_page)
    return redirect(next_page)

# render home page


@ app.route('/')
def home():
    title = 'Home'
    return render_template('index.html', title=title)


# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Fertilizer guidance'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature=float(request.form['temperature'])
        humidity=float(request.form['humidity'])
        # state = request.form.get("stt")
        # city = request.form.get("city")

        # if weather_fetch(city) != None:
            # temperature, humidity = weather_fetch(city)
        # data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        # print(f"N: {N}, P: {P}, K: {K}, pH: {ph}, Rainfall: {rainfall}")

            # Placeholder Temperature and Humidity
        # temperature = 22.61
        # humidity = 63.69

            # Preparing features array
        features = [int(N), int(P), int(K), float(temperature), float(humidity), float(ph), float(rainfall)]
        label = model.predict(crop_scaler.transform([features]))

        return render_template('crop-result.html', prediction=label[0], title=title)

    else:
        return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Fertilizer guidance'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    language = session.get('language', 'en')
    localized_guidance = PREDICTION_TRANSLATIONS.get(language, {}).get('fertilizer', {}).get(key)
    response = (
        Markup(str(fertilizer_dic[key]))
        if localized_guidance is None
        else Markup(escape(localized_guidance))
    )

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Plant health'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            predicted_label = predict_image(img)

            prediction = localized_disease_result(predicted_label)
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
     Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
     app.run(debug=False)

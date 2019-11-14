from flask import Flask, request, render_template
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import os
from io import BytesIO
import re
from base64 import b64decode
import base64


app = Flask(__name__)

pipe = tf.keras.models.load_model("model/saved_model.h5")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        base64_str = request.form['imgConverted']
        base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
        byte_data = base64.b64decode(base64_data)
        image_data = BytesIO(byte_data)

        def process_image(a):

            im = Image.open(a)
            desired_size = 200

            old_size = im.size

            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            im = im.resize(new_size, Image.ANTIALIAS)

            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))
            delta_w = desired_size - new_size[0]
            delta_h = desired_size - new_size[1]
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_im = ImageOps.expand(im, padding)

            new_im_bw = new_im.convert('L')
            new_im_bw = np.array(new_im_bw).flatten()
            new_im_bw = new_im_bw.reshape(-1, 1)
            scaler = MinMaxScaler()
            new_im_bw = scaler.fit_transform(new_im_bw)
            new_im_bw = new_im_bw.astype('float32')

            new_im_bw = new_im_bw.reshape(1, 200, 200, 1)
            return new_im_bw

        new = process_image(image_data)

        prediction = pipe.predict(new)[0]
        prediction *= 100
        pred_df = pd.DataFrame(index = ['morty', 'homer', 'rick', 'spongebob', 'pikachu', 'patrick'])
        pred_df['predictions'] = prediction

        char_label = request.form['label_char']
        char_label = char_label.lower()
        max_pred = np.argmax(prediction)
        pred_label = pred_df.index[max_pred]
        scenerio = 12
        if char_label == pred_label:
            if char_label == 'morty':
                if prediction[max_pred] >= 50:
                    scenerio =  0
                else:
                    scenerio = 1
            elif char_label == 'rick':
                if prediction[max_pred] >= 50:
                    scenerio = 2
                else:
                    scenerio = 3
            elif char_label == 'spongebob':
                if prediction[max_pred] >= 50:
                    scenerio = 4
                else:
                    scenerio = 5
            elif char_label == 'homer':
                if prediction[max_pred] >= 50:
                    scenerio = 6
                else:
                    scenerio = 7
            elif char_label == 'pikachu':
                if prediction[max_pred] >= 50:
                    scenerio = 8
                else:
                    scenerio = 9
            else:
                if prediction[max_pred] >= 50:
                    scenerio = 10
                else:
                    scenerio = 11
        else:
            if char_label == 'morty':
                scenerio = 1
            elif char_label == 'rick':
                scenerio = 3
            elif char_label == 'spongebob':
                scenerio = 5
            elif char_label == 'homer':
                scenerio = 7
            elif char_label == 'pikachu':
                scenerio = 9
            else:
                scenerio = 11
        scenerio_links = ['https://i.imgur.com/OlpSSsE.gif',
        'https://i.imgur.com/1HIRdvc.gif',
        'https://i.imgur.com/ZGtlHRA.gif',
        'https://thumbs.gfycat.com/DependentSizzlingAlligator-size_restricted.gif',
        'https://i.imgur.com/I1Brozy.gif',
        'https://i.imgur.com/S5Z7ASV.gif',
        'https://i.imgur.com/sazhW26.gif',
        'https://i.imgur.com/TeCBCeM.gif',
        'https://i.imgur.com/92f8tbq.gif',
        'https://i.imgur.com/X9LJjKz.gif',
        'https://i.imgur.com/QExCznr.gif',
        'https://i.imgur.com/gC4UWXC.gif',
        'https://freefrontend.com/assets/img/html-funny-404-pages/HTML-404-Error-Page.gif']
        test = scenerio_links[scenerio]



      #   sad face jpeg
      #   if !(arg.max pred.index == label_char):
      #     do this 6 times
      #     if char this:
      #         return this
      # else:
      #     if pred_no >= 50:
      #     do this six time
      #     if char this:
      #         return this
        return render_template('results.html', prediction=pred_df, test = test)

if __name__ == '__main__':
    app.run(debug = True, port=80)

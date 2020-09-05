import keras
from keras import backend as K
import modelCore
import tensorflow as tf
from cv2 import cv2
import numpy as np
from flask import Flask, redirect, request, render_template
import matplotlib.pyplot as plt
import base64
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)

def decode_an_image_array(rgb, dn=1):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)[:, ::dn, ::dn]
    K.clear_session()
    manTraNet = modelCore.load_trained_model()
    return manTraNet.predict(x)[0, ..., 0]


def decode_an_image_file(image_file, dn=1):
    mask = decode_an_image_array(image_file, dn)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_file[::dn, ::dn])
    plt.imshow(mask, cmap='jet', alpha=.5)
    plt.savefig('h.png', bbox_inches='tight', pad_inches=-0.1)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html", output=0)
    else:
        if 'input_image' not in request.files:
            print("No file part")
            return redirect(request.url)

        file = request.files['input_image']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            inp_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            decode_an_image_file(inp_img)
            output = cv2.imread('h.png')
            _, outputBuffer = cv2.imencode('.jpg', output)
            OutputBase64String = base64.b64encode(outputBuffer).decode('utf-8')
            return render_template("base.html", img=OutputBase64String, output=1)


if __name__ == "__main__":
    app.secret_key = 'qwertyuiop1234567890'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    

from flask import Flask, request,jsonify
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('D:/speech-emotion-recognition-master/models/cnn.keras')

d = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Neutral', 4: 'Calm', 5: 'Fearful', 6:'Disgust', 7: 'Surprised'}

def func(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=1)
    return d[predicted_label[0]]

@app.route('/', methods=['GET','POST'])
def predict():
    print("Hello World")


if __name__ == '__main__':
    # app.run(debug=True ,port=8080,use_reloader=False) 
    app.run(debug=True)


# @app.route("/")
# def home():
#     return "Home Function Called"
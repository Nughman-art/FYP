import os

from flask import Flask,request,jsonify
app = Flask(__name__)
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Conv2D, Concatenate, Flatten, Add, Dropout, GRU
from flask_cors import CORS
from flask_cors import cross_origin

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
train_data = pd.read_csv('./Final_Train_Data.csv')
test_data = pd.read_csv('./Final_Test_Data.csv')
cv_data = pd.read_csv('./Final_CV_Data.csv')
chexNet = densenet.DenseNet121(include_top=False, weights = None,   input_shape=(224,224,3), pooling="avg")
X = chexNet.output
X = Dense(14, activation="sigmoid", name="predictions")(X)
model = Model(inputs=chexNet.input, outputs=X)
if os.path.exists('./brucechou1983_CheXNet_Keras_0.3.0_weights.h5'):
    print("Weights file found.")
else:
    print("Weights file not found.")

model.load_weights('./brucechou1983_CheXNet_Keras_0.3.0_weights.h5')

chexNet = Model(inputs = model.input, outputs = model.layers[-2].output)
print("hello")
def load_image(img_name):
    image = Image.open(img_name)
    X = np.asarray(image.convert("RGB"))
    X = np.asarray(X)
    X = preprocess_input(X)
    X = resize(X, (224,224,3))
    X = np.expand_dims(X, axis=0)
    X = np.asarray(X)
    return X
encoder_model=tf.keras.models.load_model('./report_encoder_model.h5')
decoder_model = tf.keras.models.load_model('./report_decoder_model.h5')
y_train = train_data['Report']
tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(y_train.values)

def beamsearch(image, beam_width = 2):
    
    start = [tokenizer.word_index['startseq']]

    sequences = [[start, 0]]
    
    img_features = image
    img_features = encoder_model.predict(img_features)
    finished_seq = []
    
    for i in range(153):
        all_candidates = []
        new_seq = []
        for s in sequences:

            text_input = pad_sequences([s[0]], 153, padding='post')
            predictions = decoder_model.predict([text_input,img_features])
            top_words = np.argsort(predictions[0])[-beam_width:] 
            seq, score = s
            
            for t in top_words:
                candidates = [seq + [t], score - np.log(predictions[0][t])]
                all_candidates.append(candidates)
                
        sequences = sorted(all_candidates, key = lambda l: l[1])[:beam_width]
        # checks for 'endseq' in each seq in the beam
        count = 0
        for seq,score in sequences:
            if seq[len(seq)-1] == tokenizer.word_index['endseq']:
                score = score/len(seq)   # normalized
                finished_seq.append([seq, score])
                count+=1
            else:
                new_seq.append([seq, score])
        beam_width -= count
        sequences = new_seq
        
        # if all the sequences reaches its end before 155 timesteps
        if not sequences:
            break
        else:
            continue
        
    sequences = finished_seq[-1] 
    rep = sequences[0]
    score = sequences[1]
    temp = []
    rep.pop(0)
    for word in rep:
        if word != tokenizer.word_index['endseq']:
            temp.append(tokenizer.index_word[word])
        else:
            break    
    rep = ' '.join(e for e in temp)        
    
    return rep, score

@app.route("/hello",methods=["GET"])
def hello():
    if request.method=="GET":
        return "Hello Boss I am working from backend!"
@app.route("/double",methods=["POST"])
@cross_origin()
def model_double():
    if request.method=="POST":
        try:
            i1=request.files["img1"]
            i2=request.files["img2"]
            i1.save('temp1.jpg')
            i2.save("temp2.jpg")
            image = Image.open("temp1.jpg")
            X = np.asarray(image.convert("RGB"))
            X = np.asarray(X)
            X = preprocess_input(X)
            X = resize(X, (224,224,3))
            X = np.expand_dims(X, axis=0)
            X = np.asarray(X)
            img1=X
            image = Image.open("temp2.jpg")
            X = np.asarray(image.convert("RGB"))
            X = np.asarray(X)
            X = preprocess_input(X)
            X = resize(X, (224,224,3))
            X = np.expand_dims(X, axis=0)
            X = np.asarray(X)
            img2=X
            img1_features = chexNet.predict(img1)
            img2_features = chexNet.predict(img2)
            input_ = np.concatenate((img1_features, img2_features), axis=1)
            temp_input_img=input_
            rep,score=beamsearch(temp_input_img,2)
            return jsonify({"report":rep,"message":"done"}),200
        except:
            return jsonify({"message":"error"}),400
@app.route("/single",methods=["POST"])
@cross_origin()
def model_single():
    if request.method=="POST":
        print("I am entring...")
        try:
            i1=request.files["img1"]
            i1.save('temp1.jpg')
            image = Image.open("temp1.jpg")
            X = np.asarray(image.convert("RGB"))
            X = np.asarray(X)
            X = preprocess_input(X)
            X = resize(X, (224,224,3))
            X = np.expand_dims(X, axis=0)
            X = np.asarray(X)
            img1=X
            img1_features = chexNet.predict(img1)
            input_ = np.concatenate((img1_features, img1_features), axis=1)
            temp_input_img=input_
            rep,score=beamsearch(temp_input_img,2)
            return jsonify({"report":rep,"message":"done"}),200
        except:
            return jsonify({"message":"error"}),400

if __name__ == "__main__":
    cors = CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:3000"
    }
})
    app.run(debug=True)
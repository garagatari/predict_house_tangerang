import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if prediction==0:
        return render_template('index.html',
                               prediction_text='Rumah yang anda cari masuk ke dalam kategori Menengah, kategori ini memiliki rentang harga hingga 2,6 Miliar'.format(prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text='Rumah yang anda cari masuk ke dalam kategori Mewah, kategori ini memiliki rentang harga mulai dari 2,6 Miliar'.format(prediction),
                              )



if __name__ == "__main__":
    app.run(debug=True)
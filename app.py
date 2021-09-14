import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template ,make_response
import pickle
import io
import csv
from io import StringIO

app = Flask(__name__)
model = pickle.load(open('HR_Retension_Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = abs(model.predict(final_features))

    return render_template('index.html', prediction_text='HR Retention prediction : {}'.format(prediction))
@app.route('/defaults',methods=['POST'])
def defaults():
    return render_template('index.html')
@app.route('/default',methods=["POST"])
def default():
    return render_template('layout.html')
def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))
    

    # load the model from disk
    loaded_model = pickle.load(open('HR_Retension_Prediction.pkl', 'rb'))
    df['Left'] = abs(loaded_model.predict(df))

    #df = df.insert(10, 'Left', df['prediction'])

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    #response.headers["Content-Type"] = "text/csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)

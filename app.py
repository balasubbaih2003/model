from flask import Flask,request,jsonify
import pickle
import numpy as np

#http://127.0.0.1:5000/?gender=0&SSLC=88&HSC=77&SEM1=82&SEM2=86&SEM3=89
model=pickle.load(open('model4.pkl','rb'))

app=Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    gender=int(request.args['gender'])
    SSLC=int(request.args['SSLC'])
    HSC=int(request.args['HSC'])
    SEM1=int(request.args['SEM1'])
    SEM2=int(request.args['SEM2'])
    SEM3=int(request.args['SEM3'])

    pred=model.predict(np.array([[gender,SSLC,HSC,SEM1,SEM2,SEM3]])).reshape(-1,1)


    return jsonify(prediction=str(pred))

if __name__=="__main__":
    app.run(debug=True)
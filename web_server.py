import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import re
from datetime import datetime
from pandas.io.json import json_normalize

from keras.models import *
from keras.layers import *
from keras.layers.core import Lambda
from keras import backend as K

from keras.models import load_model
import keras.losses

SEQUENCE_LENGTH = 288

def q_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def gen_index(id_df, seq_length, seq_cols):
    data_matrix =  id_df[seq_cols]
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):
        yield data_matrix[stop-SEQUENCE_LENGTH:stop].values.reshape((-1,len(seq_cols)))

def reload_model():
    losses = [lambda y,f: q_loss(0.1,y,f), lambda y,f: q_loss(0.5,y,f), lambda y,f: q_loss(0.9,y,f)]

    inputs = Input(shape=(SEQUENCE_LENGTH,1))
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(inputs, training = True)
    lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.3))(lstm, training = True)
    dense = Dense(50)(lstm)
    out10 = Dense(1)(dense)
    out50 = Dense(1)(dense)
    out90 = Dense(1)(dense)
    model = Model(inputs, [out10,out50,out90])
    model.compile(loss=losses, optimizer='adam', loss_weights = [0.3,0.3,0.3])

    models = {
      'total' : keras.models.clone_model(model),
      'total_tcp' : keras.models.clone_model(model),
      'total_http' : keras.models.clone_model(model),
      'total_udp' : keras.models.clone_model(model),
      'size' : keras.models.clone_model(model),
      'size_tcp' : keras.models.clone_model(model),
      'size_http' : keras.models.clone_model(model),
      'size_udp' : keras.models.clone_model(model),
    }

    return models

def add_weights(models):
  models["total"].load_weights('saved_weights/total.h5')
  models["total_tcp"].load_weights('saved_weights/total_tcp.h5')
  models["total_http"].load_weights('saved_weights/total_http.h5')
  models["total_udp"].load_weights('saved_weights/total_udp.h5')
  models["size"].load_weights('saved_weights/size.h5')
  models["size_tcp"].load_weights('saved_weights/size_tcp.h5')
  models["size_http"].load_weights('saved_weights/size_http.h5')
  models["size_udp"].load_weights('saved_weights/size_udp.h5')
  return models

def ai_get_nn(models):
  NN = {
      'total' : K.function([models["total"].layers[0].input, K.learning_phase()], [models["total"].layers[-3].output,models['total'].layers[-2].output,models['total'].layers[-1].output]),
      'total_tcp' : K.function([models["total_tcp"].layers[0].input, K.learning_phase()], [models["total_tcp"].layers[-3].output,models['total_tcp'].layers[-2].output,models['total_tcp'].layers[-1].output]),
      'total_http' : K.function([models["total_http"].layers[0].input, K.learning_phase()], [models["total_http"].layers[-3].output,models['total_http'].layers[-2].output,models['total_http'].layers[-1].output]),
      'total_udp' : K.function([models["total_udp"].layers[0].input, K.learning_phase()], [models["total_udp"].layers[-3].output,models['total_udp'].layers[-2].output,models['total_udp'].layers[-1].output]),
      'size' : K.function([models["size"].layers[0].input, K.learning_phase()], [models["size"].layers[-3].output,models['size'].layers[-2].output,models['size'].layers[-1].output]),
      'size_tcp' : K.function([models["size_tcp"].layers[0].input, K.learning_phase()], [models["size_tcp"].layers[-3].output,models['size_tcp'].layers[-2].output,models['size_tcp'].layers[-1].output]),
      'size_http' : K.function([models["size_http"].layers[0].input, K.learning_phase()], [models["size_http"].layers[-3].output,models['size_http'].layers[-2].output,models['size_http'].layers[-1].output]),
      'size_udp' : K.function([models["size_udp"].layers[0].input, K.learning_phase()], [models["size_udp"].layers[-3].output,models['size_udp'].layers[-2].output,models['size_udp'].layers[-1].output]),
    }
  return NN
    
def prepare_dataset(df):
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
  df["total"] = pd.to_numeric(df.total, errors='coerce')
  df["total_tcp"] = pd.to_numeric(df.total_tcp, errors='coerce')
  df["total_http"] = pd.to_numeric(df.total_http, errors='coerce')
  df["total_udp"] = pd.to_numeric(df.total_udp, errors='coerce')
  df["size"] = pd.to_numeric(df.size, errors='coerce')
  df["size_tcp"] = pd.to_numeric(df.size_tcp, errors='coerce')
  df["size_http"] = pd.to_numeric(df.size_http, errors='coerce')
  df["size_udp"] = pd.to_numeric(df.size_udp, errors='coerce')
  
  df['weekday'] = df['timestamp'].dt.dayofweek
  df['H'] = df['timestamp'].dt.hour
  df['weekday_hour'] = df.weekday.astype(str) +' '+ df.H.astype(str)

  # Group by
  df['m_total'] = df.weekday_hour.replace(df.groupby('weekday_hour')['total'].mean().to_dict())
  df['m_total_tcp'] = df.weekday_hour.replace(df.groupby('weekday_hour')['total_tcp'].mean().to_dict())
  df['m_total_http'] = df.weekday_hour.replace(df.groupby('weekday_hour')['total_http'].mean().to_dict())
  df['m_total_udp'] = df.weekday_hour.replace(df.groupby('weekday_hour')['total_udp'].mean().to_dict())
  df['m_size'] = df.weekday_hour.replace(df.groupby('weekday_hour')['size'].mean().to_dict())
  df['m_size_tcp'] = df.weekday_hour.replace(df.groupby('weekday_hour')['size_tcp'].mean().to_dict())
  df['m_size_http'] = df.weekday_hour.replace(df.groupby('weekday_hour')['size_http'].mean().to_dict())
  df['m_size_udp'] = df.weekday_hour.replace(df.groupby('weekday_hour')['size_udp'].mean().to_dict())

  return df

def ai_predict(data):
  df = prepare_dataset(pd.DataFrame(data))

  total = df['total']
  total_tcp = df['total_tcp']
  total_http = df['total_http']
  total_udp = df['total_udp']
  size = df['size']
  size_tcp = df['size_tcp']
  size_http = df['size_http']
  size_udp = df['size_udp']

  m_total = df['m_total']
  m_total_tcp = df['m_total_tcp']
  m_total_http = df['m_total_http']
  m_total_udp = df['m_total_udp']
  m_size = df['m_size']
  m_size_tcp = df['m_size_tcp']
  m_size_http = df['m_size_http']
  m_size_udp = df['m_size_udp']


  total, m_total = np.log(total), np.log(m_total)
  total_tcp, m_total_tcp = np.log(total_tcp), np.log(m_total_tcp)
  total_http, m_total_http = np.log(total_http), np.log(m_total_http)
  total_udp, m_total_udp = np.log(total_udp), np.log(m_total_udp)
  size, m_size = np.log(size), np.log(m_size)
  size_tcp, m_size_tcp = np.log(size_tcp), np.log(m_size_tcp)
  size_http, m_size_http = np.log(size_http), np.log(m_size_http)
  size_udp, m_size_udp = np.log(size_udp), np.log(m_size_udp)


  total = total - m_total
  total_tcp = total_tcp - m_total_tcp
  total_http = total_http - m_total_http
  total_udp = total_udp - m_total_udp
  size = size - m_size
  size_tcp = size_tcp - m_size_tcp
  size_http = size_http - m_size_http
  size_udp = size_udp - m_size_udp


  init_total = df.m_total.apply(np.log).values
  init_total_tcp = df.m_total_tcp.apply(np.log).values
  init_total_http = df.m_total_http.apply(np.log).values
  init_total_udp = df.m_total_udp.apply(np.log).values
  init_size = df.m_size.apply(np.log).values
  init_size_tcp = df.m_size_tcp.apply(np.log).values
  init_size_http = df.m_size_http.apply(np.log).values
  init_size_udp = df.m_size_udp.apply(np.log).values


  X_total = np.reshape([total],(1,SEQUENCE_LENGTH,1))
  X_total_tcp = np.reshape([total_tcp],(1,SEQUENCE_LENGTH,1))
  X_total_http = np.reshape([total_http],(1,SEQUENCE_LENGTH,1))
  X_total_udp = np.reshape([total_udp],(1,SEQUENCE_LENGTH,1))
  X_size = np.reshape([size],(1,SEQUENCE_LENGTH,1))
  X_size_tcp = np.reshape([size_tcp],(1,SEQUENCE_LENGTH,1))
  X_size_http = np.reshape([size_http],(1,SEQUENCE_LENGTH,1))
  X_size_udp = np.reshape([size_udp],(1,SEQUENCE_LENGTH,1))


  models = reload_model()
  models = add_weights(models)

  NN = ai_get_nn(models)

  pred = {
    'total' : NN["total"]([X_total, 0.5]),
    'total_tcp' : NN["total_tcp"]([X_total_tcp, 0.5]),
    'total_http' : NN['total_http']([X_total_http, 0.5]),
    'total_udp' : NN['total_udp']([X_total_udp, 0.5]),
    'size' : NN['size']([X_size, 0.5]),
    'size_tcp' : NN['size_tcp']([X_size_tcp, 0.5]),
    'size_http' : NN['size_http']([X_size_http, 0.5]),
    'size_udp' : NN['size_udp']([X_size_udp, 0.5]),
  }

  pred = {
    'total' : np.exp(np.quantile(pred['total'][2],0.9,axis=0) + init_total[len(init_total) - len(X_total):]) - np.exp(np.quantile(pred['total'][0],0.1,axis=0) + init_total[len(init_total) - len(X_total):]),
    'total_tcp' : np.exp(np.quantile(pred['total_tcp'][2],0.9,axis=0) + init_total_tcp[len(init_total_tcp) - len(X_total_tcp):]) - np.exp(np.quantile(pred['total_tcp'][0],0.1,axis=0) + init_total_tcp[len(init_total_tcp) - len(X_total_tcp):]),
    'total_http' : np.exp(np.quantile(pred['total_http'][2],0.9,axis=0) + init_total_http[len(init_total_http) - len(X_total_http):]) - np.exp(np.quantile(pred['total_http'][0],0.1,axis=0) + init_total_http[len(init_total_http) - len(X_total_http):]),
    'total_udp' : np.exp(np.quantile(pred['total_udp'][2],0.9,axis=0) + init_total_udp[len(init_total_udp) - len(X_total_udp):]) - np.exp(np.quantile(pred['total_udp'][0],0.1,axis=0) + init_total_udp[len(init_total_udp) - len(X_total_udp):]),
    'size' :np.exp(np.quantile(pred['size'][2],0.9,axis=0) + init_size[len(init_size) - len(X_size):]) - np.exp(np.quantile(pred['size'][0],0.1,axis=0) + init_size[len(init_size) - len(X_size):]),
    'size_tcp' : np.exp(np.quantile(pred['size_tcp'][2],0.9,axis=0) + init_size_tcp[len(init_size_tcp) - len(X_size_tcp):]) - np.exp(np.quantile(pred['size_tcp'][0],0.1,axis=0) + init_size_tcp[len(init_size_tcp) - len(X_size_tcp):]),
    'size_http' : np.exp(np.quantile(pred['size_http'][2],0.9,axis=0) + init_size_http[len(init_size_http) - len(X_size_http):]) - np.exp(np.quantile(pred['size_http'][0],0.1,axis=0) + init_size_http[len(init_size_http) - len(X_size_http):]),
    'size_udp' :np.exp(np.quantile(pred['size_udp'][2],0.9,axis=0) + init_size_udp[len(init_size_udp) - len(X_size_udp):]) - np.exp(np.quantile(pred['total'][0],0.1,axis=0) + init_size_udp[len(init_size_udp) - len(X_size_udp):]),
  }

  resp = {'timestamp' : df['timestamp'][0],
          'total' : pred['total'][0],
          'total_tcp' : pred['total_tcp'][0],
          'total_http' : pred['total_http'][0],
          'total_udp' : pred['total_udp'][0],
          'size' : pred['size'][0],
          'size_tcp' : pred['size_tcp'][0],
          'size_http' : pred['size_http'][0],
          'size_udp' : pred['size_udp'][0]
  }

  print(resp)

  return resp



def resume_training(models):
  X_train = total
  y_train = df.total[SEQUENCE_LENGTH:].apply(np.log).values - init_total


  # model = reload_model()
  print("Model loaded")
  # model.load_weights(save_path)

  history = model.fit(X_train, [y_train,y_train,y_train], epochs=1, batch_size=128, verbose=2, shuffle=True)
  print("Model Trained")
  model.save_weights(save_path)
  print("Model Saved")











  














#################################################
#!flask/bin/python

from flask import Flask, jsonify,request
app = Flask(__name__)


@app.route('/', methods=['GET'])
def feedback():
    return "Done"

@app.route('/', methods=['POST'])
def receive_data():
    content = request.get_json()
    response = ai_predict(content["data"])
    return jsonify(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True,use_reloader=True)
    app.run(debug=True)
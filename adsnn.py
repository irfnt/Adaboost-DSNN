
import numpy as np
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout
from tensorflow.keras.utils import to_categorical
from keras import backend as K


my_data = np.array(pd.read_csv('/content/HTRU_2.csv', sep=',',header=None))

x = my_data[:,0:8]
y = my_data[:,8]

y = to_categorical(y)
num_classes = 2
x = preprocessing.scale(x)
#x = (x - np.mean(x)) / np.std(x) 

def new_model():
  model = Sequential()
  model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal', input_shape=(8,)))
  model.add(AlphaDropout(0.05))
  model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(128, activation='selu', kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(64, activation='selu',  kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(32, activation='selu',  kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(16, activation='selu',  kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(8, activation='selu',   kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))
  model.add(Dense(4, activation='selu',   kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.05))


#output Layer
  model.add(Dense(2, activation='softmax'))
# Compile model
  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=0.01, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model

model.fit(x, y, validation_split=0.3, epochs=10, batch_size=100, verbose=1, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(x, y, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
from SNNadaboost import AdaBoostClassifier as Ada_NN
n_estimators =3
epochs =1
bdt_real_test_NN = Ada_NN(
    base_estimator=new_model,
    n_estimators=n_estimators,
    learning_rate=0.001,
    epochs=epochs)
#######discreat:

bdt_real_test_NN.fit(x_train, y_train, batch_size)
test_real_errors_NN=bdt_real_test_NN.estimator_errors_[:]



y_pred_NN = bdt_real_test_NN.predict(X_train_r)
print('\n Training accuracy of bdt_real_test_NN (AdaBoost+DSNN): {}'.format(accuracy_score(bdt_real_test_NN.predict(X_train_r),y_train)))

y_pred_NN = bdt_real_test_NN.predict(X_test_r)
print('\n Testing accuracy of bdt_real_test_NN (AdaBoost+DSNN): {}'.format(accuracy_score(bdt_real_test_CNN.predict(X_test_r),y_test)))

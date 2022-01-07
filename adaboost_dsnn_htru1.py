import pandas as pd 
import numpy as np
from tensorflow.keras import models
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
# In this code I am going to use keras to classify a binary output
#First we need to load the data scale it and define our training data
#Then I will define a network of layers that amps my input to my targets
#Then I will need a loss funtion and some kind of metric to understand the learning process
#Finally I will iterate through the training data using fit()

df=pd.read_csv('/content/all_in_one.csv')
#Load the data from the HTRU csv and print its hsape

df.shape

df.head()

#df = pd.DataFrame.to_numpy(pulsar_data)
# It seems there are NaNs in the dataset.. 
# And all of them are in column 3. The one with log(.)
#df = df[~np.isnan(df).any(axis=1)]
#X = df[:,0:6] # Our feature matrix
#y = np.array(df[:,-1],dtype=np.int) # Our truth labels
#idx = np.array(df[:,6],dtype=np.int) # our ID
#clname = ['RFI','PULSAR']
# Whitening
#X -= np.mean(X,axis=0)
#X /= np.std(X,axis=0)
#df1 = df[~np.isnan(df).any(axis=1)]
df1=df.dropna(subset=['three'])
df1.head()

df1.isnull().sum()

df2=df1.drop('id', axis=1)
df2.head()

#x = df1.iloc[:,0:7]
#y = df1.iloc[:,6]

#Seperate the x variables and the y varaibles
#np.random.seed(10) 
#X_MinMax = preprocessing.MinMaxScaler(x)
#scale the x variables
X = df2.drop('yn',axis=1)
y = df2['yn']
#reshape the x and y variables

smote=SMOTE (random_state=12, sampling_strategy='minority')
x_sm, y_sm=smote.fit_resample(X,y)
y_sm1=pd.Series(y_sm)
y_sm1.value_counts()

y11 = to_categorical(y_sm)
num_classes = 2
x11 = preprocessing.scale(x_sm)

X_train, X_test, Y_train, Y_test = train_test_split(x_sm, y_sm1, test_size=0.2,random_state=15, stratify=y_sm1)
#here I define my training data and split it 20% off of it to create the test data
X_train.shape

Y_test.shape

scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test.shape

import tensorflow as tf

def new_DSNN():
  model= Sequential()
  #model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal', input_dim=6,))
  #model.add(AlphaDropout(0.05))
  #model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal'))
  #model.add(AlphaDropout(0.05))
  #model.add(Dense(128, activation='selu', kernel_initializer='lecun_normal'))
  #model.add(AlphaDropout(0.05))
  #model.add(Dense(64, activation='selu',  kernel_initializer='lecun_normal'))
  #model.add(AlphaDropout(0.05))
  #model.add(Dense(32, activation='selu',  kernel_initializer='lecun_normal'))
  #model.add(AlphaDropout(0.05))
  model.add(Dense(16, activation='selu',  kernel_initializer='lecun_normal',input_dim=6,))
  model.add(AlphaDropout(0.005))
  model.add(Dense(8, activation='selu',   kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.005))
  model.add(Dense(4, activation='selu',   kernel_initializer='lecun_normal'))
  model.add(AlphaDropout(0.005))
  model.add(Dense(1, activation='sigmoid'))
  adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=0.01, decay=0.0)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model=new_DSNN()

history= model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), verbose=1)

!pip install SciencePlots
import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')

! sudo apt-get install texlive-latex-recommended 
! sudo apt install texlive-latex-extra
! sudo apt install dvipng

! sudo apt install cm-super

print(history.history.keys())
# summarize history for accuracy
plt.rcParams.update({'font.size': 10})
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig('Model Accuracy', dpi=400)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.legend(['Training', 'Validation'], loc='uper right')
plt.savefig('Model Loss', dpi=400)
plt.show()

class MyKerasClassifier(KerasClassifier):
  def fit(self, X_train, Y_train, sample_weight=None, **kwargs):
    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
    # Arguments
        x : array-like, shape `(n_samples, n_features)`
            Training samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.fit`
    # Returns
        history : object
            details about the training history at each epoch.
    # Raises
        ValueError: In case of invalid shape for `y` argument.
    """
    y = np.array(Y_train)
    if len(y.shape) == 2 and y.shape[1] > 1:
        self.classes_ = np.arange(y.shape[1])
    elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
    else:
        raise ValueError('Invalid shape for y: ' + str(y.shape))
    self.n_classes_ = len(self.classes_)
    if sample_weight is not None:
        kwargs['sample_weight'] = sample_weight
    return super(MyKerasClassifier, self).fit(X_train, Y_train, **kwargs)
    #return super(KerasClassifier, self).fit(x, y, sample_weight=sample_weight)

  def predict(self, X_train, **kwargs):
    """Returns the class predictions for the given test data.
    # Arguments
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        **kwargs: dictionary arguments
            Legal arguments are the arguments
            of `Sequential.predict_classes`.
    # Returns
        preds: array-like, shape `(n_samples,)`
            Class predictions.
    """
    kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
    classes = self.model.predict_classes(X_train, **kwargs)
    return self.classes_[classes].flatten()

from sklearn.base import BaseEstimator, ClassifierMixin
class BinaryClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self.model = new_DSNN()
    
    self.classifier = MyKerasClassifier(build_fn=new_DSNN, epochs=100,validation_data=(X_test, Y_test), batch_size=32, verbose=1)

  def fit(self, X_train, Y_train, sample_weight=None):
    self.n_classes_ = 2
    self.classes_ = np.array([0, 1])
    self.classifier.fit(X_train, Y_train, sample_weight)

  def predict(self, X_train):
    prediction = self.classifier.predict(X_train)
    return prediction

from sklearn.ensemble import AdaBoostClassifier as ad

boosted_classifier = ad(
    base_estimator=BinaryClassifier(),
    n_estimators=1,
    random_state=0,
    learning_rate=0.001,
    algorithm= 'SAMME')

boosted_classifier.fit(X_train,Y_train)



print(history.history.keys())
# summarize history for accuracy
plt.rcParams.update({'font.size': 9})
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Epoch',fontsize=10)
plt.legend(['Training', 'validation'], loc='lower right')
plt.savefig('Model Accuracy', dpi=400)
plt.show()
# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss', fontsize=10)
plt.xlabel('Epoch',fontsize=12)
plt.legend(['Training', 'Validation'], loc='uper right')
plt.savefig('Model Loss', dpi=400)
plt.show()

y_pred=boosted_classifier.score(X_test, Y_test)
print(y_pred)

y_pred1=boosted_classifier.predict(X_test)

from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

print(f"CLASSIFICATION REPORT:\n\n{classification_report(Y_test,y_pred1)}")

print(confusion_matrix(Y_test,y_pred1))

label_aux = plt.subplot()
cm = confusion_matrix(Y_test, np.round(y_pred1))
cm_nn = pd.DataFrame(cm, index = ['Non-Pulsars','Pulsars'], columns = ['Non-Pulsars','Pulsar'])
sns.heatmap(cm_nn,annot=True,fmt="d", cbar=False)

label_aux.set_xlabel('Predicted Value');label_aux.set_ylabel('True Value');
plt.savefig('Confussion Matrix of HTRU-1 adaboost-DSNN_PRC',dpi=400)

from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
ptbl = PrettyTable()
ptbl.field_names = ["Accuracy", "Recall","precision","F1Score"]
ptbl.add_row([accuracy_score(Y_test,np.round(y_pred1)),recall_score(Y_test, np.round(y_pred1)), 
              precision_score(Y_test,np.round(y_pred1)),f1_score(Y_test, np.round(y_pred1))])
print(ptbl)

from sklearn import model_selection, metrics
from sklearn.metrics import precision_recall_curve
plt.rcParams.update({'font.size': 10})
Precision, Recall, thresholds = precision_recall_curve(Y_test, y_pred1)
plt.plot(Recall, Precision, label='Adaboost-DSNN (area = %0.1f)')
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)
plt.savefig('adaboost-DSNN_PRC',dpi=400)
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
plt.rcParams.update({'font.size': 10})
logit_roc_auc = roc_auc_score(Y_test, boosted_classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, boosted_classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Adaboost-DSNN (area = %0.1f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.legend(loc="lower right")
plt.savefig('adaboost-DSNN_ROC',dpi=400)
plt.show()

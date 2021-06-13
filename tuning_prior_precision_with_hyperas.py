"""

Sample hyperas script for tuning prior precision

data() and model() functions must be created for Hyperas' use
because it wraps Hyperopt

"""

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras import backend as K
from . import Vadam

if K.backend() == 'tensorflow':
    import tensorflow as tf

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


def data():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    return data, labels, val_data, val_labels


def model(data, labels, val_data, val_labels):

    model = Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Prior precision lower and upper bounds are from appendix I.1 tests

    optimizer = Vadam(lr={{uniform(1e-4, 1e-2)}},
                      prior_prec={{uniform(1e-2, 25)}},
                      train_set_size=1000)

    model.compile(loss='kld',
                  metrics=['mae'],
                  optimizer=optimizer)

    result = model.fit(data,
                       labels,
                       epochs=10,
                       batch_size=32,
                       validation_data=(val_data, val_labels))

     # get the lowest validation loss of each training epoch
     validation_loss = np.amin(result.history['val_loss'])
     print('Best loss of epoch:', validation_loss)

     return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=100,
                                      trials=Trials(),
                                      eval_space=True)

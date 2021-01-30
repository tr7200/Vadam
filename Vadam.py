from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import six
from six.moves import zip

import numpy as np

from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras import backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class Vadam(Optimizer):
    """variational bayes Adam optimizer, same params as Keras Adam with
    
    ARGS:
        - train_set_size (int): training dataset size
        - prior_prec (float): prior precision (1.0 is uninformative prior)
        - prec_init (float): precision
        
    RETURNS:
        weight-perturbed gradient updates 
    """
    def __init__(self, 
                 learning_rate=0.001,
                 train_set_size=train_set_size,
                 beta_1=0.9, 
                 beta_2=0.999,
                 prior_prec=1.0, 
                 prec_init=1.0,
                 **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        
        learning_rate = kwargs.pop('lr', learning_rate)
        train_set_size = kwargs.pop('train_set_size', train_set_size)
        prior_prec = kwargs.pop('prior_prec', prior_prec)
        prec_init = kwargs.pop('prec_init', prec_init)
        
        super(Vadam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.train_set_size = train_set_size
            self.prior_prec = prior_prec
            self.prec_init = prec_init

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]
                
        vhats = [K.zeros(1, name='vhat_' + str(i))
                 for i in range(len(params))]
       
        self.weights = [self.iterations] + ms + vs + vhats
        
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g + /
                ((self.prior_prec * p)/self.train_set_size)
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            
            # bias correction, caused numerical instability
            #m_t = m_t  / (1. - self.beta_1) 
            #v_t = v_t / (1. - self.beta_2)
            
            p_t = (p + self.epsilon) - lr_t * m_t / (K.sqrt(v_t) + /
                 self.prior_prec/self.train_set_size)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
          
            return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'train_set_size': self.train_set_size,
                  'prior_prec': self.prior_prec,
                  'prec_init': self.prec_init}
        base_config = super(Vadam, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

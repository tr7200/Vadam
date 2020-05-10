# Vadam

Keras optimizer that modified Adam to approximate variational inference 
following [arXiv 1712.07628](https://arxiv.org/abs/1806.04854).

    Khan, M. E., Nielsen, D., Tangkaratt, V., Lin, W., Gal, Y., 
    & Srivastava, A. (2018). Fast and scalable bayesian deep 
    learning by weight-perturbation in adam. 
    arXiv preprint arXiv:1806.04854.
       
- This optimizer supports Keras 2.3.1 since the Tensorflow 2.0 version of
Adam separates gradients by sparsity, and this algorithm does not support 
sparse gradients according to the authors' Pytorch implementation.
- The default prior precision value (Lambda in the paper) results in a 
completely uninformative prior that will *NOT* yield viable results by the 
authors' own admission. According to the relevant section of the paper, 
finding the right value of Lambda is beyond the scope of the paper but an 
example Hyperas script that tunes Adam simultaneously on learning rate and 
prior precision is offered here to address this.
- This verision of the Vadam algorithm follows slides 11 of 15 from the 
2018 ICML presentation [slides](https://goo.gl/ouZRkM), which is slightly 
different from the paper. In this version, the Keras backend fuzz factor is 
used instead of Monte Carlo sampling of the gradients. See 
[here](https://thinklab.com/content/2693564) for more information.
       
Usage (only required parameter is train_set_size, though prior_prec should 
be tuned):

    import numpy as np
    X_train = np.random.random((1000, 32))
    Y_train = np.random.random((1000, 10))
    
    model = Sequential()
    ...    
    model.compile(optimizer=Vadam(train_set_size=1000,
                          ...)
     
    result = model.fit(X_train,
                       Y_train,
                       ...)

Tensorflow < 2.0 for the reason described above, made with Keras 2.3.1 but it
should work with 2.2.4.

This callback is more suitable for training with image or text data for hundreds of 
epochs.

`python setup.py install` to install.

MIT License

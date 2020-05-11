# Vadam

Keras optimizer that modifies the Adam optimizer to approximate variational 
inference by perturbing weights following 
[arXiv 1712.07628](https://arxiv.org/abs/1806.04854).

    Khan, M. E., Nielsen, D., Tangkaratt, V., Lin, W., Gal, Y., 
    & Srivastava, A. (2018). Fast and scalable bayesian deep 
    learning by weight-perturbation in adam. 
    arXiv preprint arXiv:1806.04854.
       
- This optimizer supports Keras 2.3.1 since the Tensorflow 2.0 version of
Adam separates gradients by sparsity, and this algorithm does not support 
sparse gradients according to the authors' Pytorch implementation.
- The default prior precision value (Lambda in the paper) results in a 
completely uninformative prior that will *NOT* yield viable results by the 
authors' own admission (appendix K.3). According to the relevant section 
of the paper, finding the right value of Lambda is beyond the scope of the 
paper but an example Hyperas script that tunes Vadam simultaneously on 
learning rate and prior precision is offered here to address this.
- This verision of the Vadam algorithm follows slides 11 of 15 from the 
2018 ICML presentation [slides](https://goo.gl/ouZRkM), which is slightly 
different from the paper. In this implementation of the version 
of the algorithm from the slides, only the epsilon fuzz factor is added to 
parameter updates instead of mean and standard deviations derived from a 
diagonal multivariate gaussian distribution, though those may be added 
in the future.
- The Pytorch version of Vadam also includes the ability to provide Monte 
Carlos sampling to parameter updates, which is not included here. However, 
the ablation tests in appendix I.2 uses 1 Monte Carlo sample so this 
simplification may not adversely affect variation too badly. 
See [here](https://thinklab.com/content/2693564) for more information on the 
presentation.
- Unlike the Keras implementation of Adam, both the Pytorch implementation 
of Adam as well as Vadam perform bias correction. Bias correction is 
therefore added here as well, but using it resulted in numerical instability 
and code is left commented out.
- The Adagrad option is removed since it is not in the Pytorch 
implementation.
       
Usage (only required parameter is train_set_size, though prior_prec should 
definitely be tuned):

    import numpy as np
    X_train = np.random.random((1000, 32))
    Y_train = np.random.random((1000, 10))
    
    model = Sequential()
    ...    
    model.compile(optimizer=Vadam(train_set_size=1000,
                          ...)
    
    # train_set_size parameter is from X_train
    
    result = model.fit(X_train,
                       Y_train,
                       ...)

Only works with Tensorflow < 2.0 for the reason described above, made with 
Keras 2.3.1 but it should work with 2.2.4.

This optimizer is suitable for approximating variational inference in a 
neural network to provide probablistic output that provide upper and 
lower confidence bounds on prediction.

MIT License

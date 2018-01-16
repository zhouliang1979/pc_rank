from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import random
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lstm import *



def predict(
    dim_proj=18,  # word embeding dimension and LSTM number of hidden units.
    dim_finalx=15,
    dim_hiden=10,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=50,  # The maximum number of epoch to run
    dispFreq=10000,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    validFreq=10000,  # Compute the validation error after this number of update.
    saveFreq=10000,  # Save the parameters after every saveFreq updates
    maxlen=20,  # Sequence longer then this get ignored
    batch_size=20,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',
    test_data_path='balance_dataset_MoreHigher.txt',
    model_path='lstm_model_MoreHigher_regularized_2e-4.npz',
    prob_path='prob_lstm',
    # Parameter for extra option
    noise_std=0.3,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=True,  # Path to a saved model we want to start from.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    test = load_data(test_data_path, maxlen=maxlen)

    model_options['ydim1'] = 7
    model_options['ydim2'] = 6
    model_options['ydim3'] = 2
    model_options['ydim4'] = 2
    model_options['ydim5'] = 2

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params(model_path, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y1, y2, y3, y4, y5, final_x,
     f_pred_prob1,  f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,
     f_pred1, f_pred2, f_pred3, f_pred4, f_pred5,
     cost, cost_reward) = build_model(tparams, model_options)

    prob_file = open(prob_path, "w")
    uidx = 0
    kf = get_minibatches_idx(len(test[0]), batch_size, shuffle=False)
    test_size = len(test[0])
    test_correct = 0
    for _, train_index in kf:
        if uidx%1000==0:
            print(uidx)
        uidx += 1
        use_noise.set_value(1.)

        # Select the random examples for this minibatch
        y = [test[1][t] for t in train_index]
        x = [test[0][t] for t in train_index]

        # Get the data in numpy.ndarray format
        # This swap the axis!
        # Return something of shape (minibatch maxlen, n samples)

        x, mask, y1, y2, y3, y4, y5, final_x = prepare_data(x, y)

        pred_prob1 = f_pred_prob1(x, mask)
        preds1 = f_pred1(x, mask)
        test_correct += (preds1 == y1).sum()
        x = x.swapaxes(0, 1)
        mask = mask.swapaxes(0, 1)
        
        for _prob1, _label, _f, _m  in zip(pred_prob1, y1, x, mask):
            prob_file.write(str(_label+1))
            prob_file.write("\t")
            prob_file.write((','.join([str(i) for i in _prob1])))
            prob_file.write("\t")

            f_str = []
            for step, m_i in zip(_f, _m):
                if m_i == 0.0:
                    break
                step_str = []
                for index, value in enumerate(step):
                    if value ==1 :
                        if index <= 7:
                            step_str.append(str(index))
                        elif index <= 14:
                            step_str.append(str(index-8))
                   
                step_str.append(str(int(step[15])) + str(int(step[16]))  + str(int(step[17])) )
                f_str.append(','.join(step_str))
            prob_file.write('#'.join(f_str) + '\n')
    prob_file.close()

    print("test error:", 1.0 - float(test_correct)/test_size)
    return  0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("check_ProbDistr.py test_set model_bin prob_path dim_hiden")
    else:
        print("start test:")
    predict(
        test_data_path=sys.argv[1] ,
        model_path=sys.argv[2],  
        prob_path=sys.argv[3],
        dim_hiden=int(sys.argv[4]),
    )




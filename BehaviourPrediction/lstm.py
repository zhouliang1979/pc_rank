
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

import dataset_allinone as dataset

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # Make a minibatch out of what is left
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def zipp(params, tparams):
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped, model_path):
    print ("unzip param_file")
    param_file = open(model_path, 'w')
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        print(kk)
        aa = vv.get_value()
        new_params[kk] = aa
        param_file.write(kk)
        param_file.write('\n')
        if 'b' not in kk:
            print(len(aa), len(aa[0]))
            for i in range(0,len(aa[0])):
                values = []
                for j in range(0,len(aa)):
                    value = aa[j, i]
                    values.append(str(value))
                param_file.write(','.join((values)))
                param_file.write('\n')
        else:
            print(len(aa))
            values = []
            for value in aa:
                values.append(str(value))
            param_file.write(','.join((values)))
            param_file.write('\n')

    param_file.close()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


#Global (not LSTM) parameter. For the embeding and the classifier.
def init_params(options):
    params = OrderedDict()

    params = param_init_lstm(options, params, prefix=options['encoder'])

    # classifier
    params['W1'] = 0.01 * numpy.random.randn(options['dim_hiden'],
                                             options['ydim1']).astype(config.floatX)
    params['b1'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)

    params['W2'] = 0.01 * numpy.random.randn(options['dim_hiden'],
                                             options['ydim2']).astype(config.floatX)
    params['b2'] = numpy.zeros((options['ydim2'],)).astype(config.floatX)

    params['W3'] = 0.01 * numpy.random.randn(options['dim_hiden'] + options['dim_finalx'],
                                             options['ydim3']-1).astype(config.floatX)
    params['b3'] = numpy.zeros((options['ydim3']-1,)).astype(config.floatX)

    params['W4'] = 0.01 * numpy.random.randn(options['dim_hiden'] + options['dim_finalx'],
                                             options['ydim4']-1).astype(config.floatX)
    params['b4'] = numpy.zeros((options['ydim4']-1,)).astype(config.floatX)

    params['W5'] = 0.01 * numpy.random.randn(options['dim_hiden'] + options['dim_finalx'],
                                             options['ydim5']-1).astype(config.floatX)
    params['b5'] = numpy.zeros((options['ydim5']-1,)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams



def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    params[_p(prefix, 'U')] = 0.01 * numpy.random.randn(options['dim_hiden'], options['dim_hiden']*4).astype(config.floatX)
    params[_p(prefix, 'W')] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_hiden']* 4).astype(config.floatX)
    params[_p(prefix, 'b')] = numpy.zeros((options['dim_hiden']*4, )).astype(config.floatX)

    return params

#state_below is three-dimensional tensor,  nsteps * nsamples * dim_proj
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_hiden']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_hiden']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_hiden']))
        c = tensor.tanh(_slice(preact, 3, options['dim_hiden']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           options['dim_hiden']),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           options['dim_hiden'])],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def adadelta(lr, tparams, grads, x, mask, y1, y2, y3, y4, y5,  final_x, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y1, y2, y3, y4, y5, final_x], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype='float64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y1 = tensor.vector('y1', dtype='int64')
    y2 = tensor.vector('y2', dtype='int64')
    y3 = tensor.vector('y3', dtype='int64')
    y4 = tensor.vector('y4', dtype='int64')
    y5 = tensor.vector('y5', dtype='int64')
    final_feature = tensor.matrix('final_feature', dtype='float64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    proj = lstm_layer(tparams, x, options, prefix=options['encoder'], mask=mask)

    if options['encoder'] == 'lstm':
        proj = proj[-1]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred1 = tensor.nnet.softmax(tensor.dot(proj, tparams['W1']) + tparams['b1'])
    pred2 = tensor.nnet.softmax(tensor.dot(proj, tparams['W2']) + tparams['b2'])

    proj_2 = tensor.concatenate([proj, final_feature], axis=1)
    pred_3_1 = tensor.nnet.sigmoid(tensor.dot(proj_2, tparams['W3']) + tparams['b3'])
    pred_4_1 = tensor.nnet.sigmoid(tensor.dot(proj_2, tparams['W4']) + tparams['b4'])
    pred_5_1 = tensor.nnet.sigmoid(tensor.dot(proj_2, tparams['W5']) + tparams['b5'])
    pred_3_0 = 1.0 - pred_3_1
    pred_4_0 = 1.0 - pred_4_1
    pred_5_0 = 1.0 - pred_5_1
    pred3 = tensor.concatenate([pred_3_0, pred_3_1], axis=1)
    pred4 = tensor.concatenate([pred_4_0, pred_4_1], axis=1)
    pred5 = tensor.concatenate([pred_5_0, pred_5_1], axis=1)
  
    f_pred_prob1 = theano.function([x, mask], pred1, name='f_pred_prob1')
    f_pred1 = theano.function([x, mask], pred1.argmax(axis=1), name='f_pred1')

    f_pred_prob2 = theano.function([x, mask], pred2, name='f_pred_prob2')
    f_pred2 = theano.function([x, mask], pred2.argmax(axis=1), name='f_pred2')

    f_pred_prob3 = theano.function([x, mask, final_feature], pred3, name='f_pred_prob3')
    f_pred3 = theano.function([x, mask, final_feature], pred3.argmax(axis=1), name='f_pred3')

    f_pred_prob4 = theano.function([x, mask, final_feature], pred4, name='f_pred_prob4')
    f_pred4 = theano.function([x, mask, final_feature], pred4.argmax(axis=1), name='f_pred4')

    f_pred_prob5 = theano.function([x, mask, final_feature], pred5, name='f_pred_prob5')
    f_pred5 = theano.function([x, mask, final_feature], pred5.argmax(axis=1), name='f_pred5')



    off = 1e-8
    if pred3.dtype == 'float16':
        off = 1e-6

    cost1 = -tensor.log(pred1[tensor.arange(n_samples), y1] + off).mean()
    cost2 = -tensor.log(pred2[tensor.arange(n_samples), y2] + off).mean()
    cost3 = -tensor.log(pred3[tensor.arange(n_samples), y3] + off).mean()
    cost4 = -tensor.log(pred4[tensor.arange(n_samples), y4] + off).mean()
    cost5 = -tensor.log(pred5[tensor.arange(n_samples), y5] + off).mean()

    cost = cost1 + cost2 + cost3 + cost4 + cost5
    cost_reward = cost3 + cost4 + cost5
    return use_noise, x, mask, y1, y2, y3, y4, y5, final_feature, \
           f_pred_prob1, f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,\
           f_pred1, f_pred2, f_pred3, f_pred4, f_pred5, cost, cost_reward


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred1, f_pred2, f_pred3, f_pred4, f_pred5, prepare_data, data, iterator,  f_cost):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err1 = 0
    valid_err2 = 0
    valid_err3 = 0
    valid_err4 = 0
    valid_err5 = 0

    total_num = 0
    total_cost = 0
    
    
    precision = [0,0,0,0,0,0,0,0]
    predict_num = [0,0,0,0,0,0,0,0]
    class_num = [0,0,0,0,0,0,0,0]
    for _, valid_index in iterator:
        x, mask, y1, y2, y3, y4, y5, final_x = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        total_num += len(valid_index)
        preds1 = f_pred1(x, mask)
        valid_err1 += (preds1 == y1).sum()
        for i1, i2 in zip(preds1, y1):
            class_num[int(i2)]  += 1
            predict_num[int(i1)] += 1
            if(i1 == i2):
                precision[int(i2)] += 1
        preds2 = f_pred2(x, mask)
        valid_err2 += (preds2 == y2).sum()
        preds3 = f_pred3(x, mask, final_x)
        valid_err3 += (preds3 == y3).sum()
        preds4 = f_pred4(x, mask, final_x)
        valid_err4 += (preds4 == y4).sum()
        preds5 = f_pred5(x, mask, final_x)
        valid_err5 += (preds5 == y5).sum()
        total_cost += f_cost(x, mask, y3, y4, y5, final_x)

    valid_err1 = 1. - numpy_floatX(valid_err1) / total_num
    valid_err2 = 1. - numpy_floatX(valid_err2) / total_num
    valid_err3 = 1. - numpy_floatX(valid_err3) / total_num
    valid_err4 = 1. - numpy_floatX(valid_err4) / total_num
    valid_err5 = 1. - numpy_floatX(valid_err5) / total_num
    cost = total_cost/len(iterator)

    return (valid_err1, valid_err2, valid_err3, valid_err4, valid_err5, cost)


def train_lstm(
    dim_proj=18,  # word embeding dimension and LSTM number of hidden units.
    dim_finalx=15,
    dim_hiden=10,
    train_dataset='balance_dataset_MoreHigher.txt',
    test_dataset='balance_dataset_MoreHigher.txt',
    max_epochs=50,  # The maximum number of epoch to run
    decay_c=0.0002,  # Weight decay for the classifier applied to the weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model_MoreHigher_regularized_2e-4.npz',  # The best model will be saved there
    model_path='param_file_2e-4',
    validFreq=1000,  # Compute the validation error after this number of update.
    saveFreq=1000,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=200,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.

    # Parameter for extra option
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    print('Loading train data:', train_dataset)
    train = dataset.load_data(train_dataset, maxlen)
    print('Loading test data:', test_dataset)
    test  = dataset.load_data(test_dataset, maxlen)

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
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y1, y2, y3, y4, y5, final_x,
     f_pred_prob1,  f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,
     f_pred1, f_pred2, f_pred3, f_pred4, f_pred5,
     cost, cost_reward) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['lstm_W'] ** 2).sum() + (tparams['lstm_U'] ** 2).sum() + (tparams['lstm_b'] ** 2).sum()
        weight_decay *= 0.5
        weight_decay += (tparams['W1'] ** 2).sum() + (tparams['b1'] ** 2).sum()
        weight_decay += (tparams['W2'] ** 2).sum() + (tparams['b2'] ** 2).sum()
        weight_decay += (tparams['W3'] ** 2).sum() + (tparams['b3'] ** 2).sum()
        weight_decay += (tparams['W4'] ** 2).sum() + (tparams['b4'] ** 2).sum()
        weight_decay += (tparams['W5'] ** 2).sum() + (tparams['b5'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost_reward = theano.function([x, mask, y3, y4, y5, final_x], cost_reward, name='f_cost_reward')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y1, y2, y3, y4, y5, final_x], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y1, y2, y3, y4, y5, final_x, cost)

    print('Optimization')

    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in range(max_epochs):
            n_samples = 0
            errors1 = 0
            errors2 = 0
            errors3 = 0
            errors4 = 0
            errors5 = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            train_size = len(train[0])
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y1, y2, y3, y4, y5, final_x= dataset.prepare_data(x, y)
              
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y1, y2, y3, y4, y5, final_x)

                preds1 = f_pred1(x, mask)
                preds2 = f_pred2(x, mask)
                preds3 = f_pred3(x, mask, final_x)
                preds4 = f_pred4(x, mask, final_x)
                preds5 = f_pred5(x, mask, final_x)
                
                errors1 += (preds1 != y1).sum()
                errors2 += (preds2 != y2).sum()
                errors3 += (preds3 != y3).sum()
                errors4 += (preds4 != y4).sum()
                errors5 += (preds5 != y5).sum()
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    test_err = pred_error(f_pred1, f_pred2, f_pred3, f_pred4, f_pred5, dataset.prepare_data, test, kf_test, f_cost_reward)
                    print('test error rate', test_err)


            print('eidx:%d of epoch:%d the train error is (%f, %f, %f %f, %f) , cost is %f' %
                  ( eidx + 1, max_epochs, float(errors1)/n_samples, float(errors2)/n_samples, float(errors3)/train_size,
                  float(errors4)/train_size,float(errors5)/train_size, cost))

            end_time = time.time()
            print('The code run for %d epochs,took %.1fs,  with %f sec/epochs, remain %f hour' % (
                (eidx + 1), (end_time - start_time),  (end_time - start_time) / (1. * (eidx + 1)), 
                (end_time - start_time) * (max_epochs-eidx-1)/ (3600. * (eidx + 1))))

    except KeyboardInterrupt:
        print("Training interupted")

    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams, model_path)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred1, f_pred2, f_pred3, f_pred4, f_pred5,  dataset.prepare_data, train, kf_train_sorted, f_cost_reward)
    test_err = pred_error(f_pred1, f_pred2, f_pred3, f_pred4, f_pred5, dataset.prepare_data, test, kf_test, f_cost_reward)

    print('Train error rate', train_err)
    print('Test error rate', test_err )
    if saveto:
        numpy.savez(saveto, train_err=train_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    
    end_time = time.time()
    print('The code run for %d epochs,took %.1fs,  with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time),  (end_time - start_time) / (1. * (eidx + 1))))


if __name__ == '__main__':
    
    if len(sys.argv) != 6:
        print("lstm.py train test model_txt model_bin dim_hiden")
    else:
        print("start training:")
    train_lstm(
        max_epochs=100,
        batch_size=200, 
        train_dataset=sys.argv[1] ,
        test_dataset=sys.argv[2] ,
        model_path=sys.argv[3] ,
        saveto=sys.argv[4] , 
        dim_hiden=int(sys.argv[5]),
    )


from keras.optimizers import Adam
from keras.utils import generic_utils
from ops import *
import keras.backend as K
import sys
# Utils
sys.path.append("utils/")
from simple_utils import plot_batch_train, plot_batch_eval

import os
import time


def feature_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def pixel_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def variation_loss(y_true, y_pred):
    # Assume img size is 64*64
    if K.image_dim_ordering() == 'tf':
        a = K.square(y_pred[:, :64-1, :64-1, :] - y_pred[:, 1:, :64-1, :])
        b = K.square(y_pred[:, :64-1, :64-1, :] - y_pred[:, :64-1, 1:, :])
    else:
        a = K.square(y_pred[:, :, 64 - 1, :64 - 1] - y_pred[:, :, 1:, :64 - 1])
        b = K.square(y_pred[:, :, 64 - 1, :64 - 1] - y_pred[:, :, :64 - 1, 1:])
    return K.sum(K.sqrt(a+b))


def train(batch_size, n_batch_per_epoch, nb_epoch, sketch, color, weights, tag, sk_ir, save_weight=1, img_dim=[64,64,1]):
    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model, model_name = edge2color(img_dim, batch_size=batch_size)

    model.compile(loss=[pixel_loss, feature_loss], loss_weights=[1, 1], optimizer=opt)
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model, to_file='figures/edge2color.png', show_shapes=True, show_layer_names=True)

    global_counter = 1
    for epoch in range(nb_epoch):
        batch_counter = 1
        start = time.time()
        batch_idxs = sketch.shape[0] // batch_size
        if n_batch_per_epoch >= batch_idxs or n_batch_per_epoch == 0:
            n_batch_per_epoch = batch_idxs
        progbar = generic_utils.Progbar(n_batch_per_epoch * batch_size)

        sk_val = sketch[0:16]
        co_val = color[0:16]
        sketch = sketch[16:]
        color = color[16:]
        weights = weights[16:]

        for idx in range(batch_idxs):
            batch_sk = sketch[idx * batch_size: (idx + 1) * batch_size]
            batch_co = color[idx * batch_size: (idx + 1) * batch_size]
            batch_weights = weights[idx * batch_size: (idx + 1) * batch_size]
            train_loss = model.train_on_batch([batch_sk], [batch_co, batch_weights])
            batch_counter += 1
            progbar.add(batch_size, values=[('pixel_loss', train_loss[1]), ('feature_loss', train_loss[2])])
            # if batch_counter >= n_batch_per_epoch:
            if global_counter % 50 == 1:
                plot_batch_train(model, img_dim[0], batch_size, batch_sk, batch_co, epoch, idx, tag)
                plot_batch_eval(model, img_dim[0], batch_size, sk_val, tag=tag+'_val')
                plot_batch_eval(model, img_dim[0], batch_size, sk_ir, tag=tag+'_test')
            global_counter += 1

            if batch_counter >= n_batch_per_epoch:
                break
        print ""
        print 'Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start)

        if save_weight:
            # save weights every epoch
            weights_path = '%s/%s_weights_epoch_%s.h5' % (model_name, tag, epoch)
            if not os.path.exists('%s' % model_name):
                os.mkdir('%s' % model_name)
            model.save_weights(weights_path, overwrite=True)


def evaluate(batch_size, tag, epoch, sketch, img_dim=[64,64,1]):
    model, model_name = edge2color(img_dim, batch_size=batch_size)
    model.load_weights('%s/%s_weights_epoch_%s.h5' % (model_name, tag, epoch))
    print 'Load Model Complete'
    plot_batch_eval(model, img_dim[0], batch_size=batch_size, sketch=sketch, tag=tag)


if __name__ == '__main__':
    sketch, color, weights, sk_ir = load_with_size(os.path.expanduser
                                   ('~/Desktop/hdf5/clear/color_ir_sketch.h5'), img_size=128)
                                 # ('~/Desktop/hdf5/clear/database_train.h5'))
                                 # ('~/Desktop/DeepLearningImplementations/DFI/data/processed/lfw_224_data.h5'))
    img_dim = [128, 128, 1]
    tag = '1-1-4batch-128'
    train(batch_size=16, n_batch_per_epoch=4, nb_epoch=10000, sketch=sketch, color=color,
          weights=weights, tag=tag, sk_ir=sk_ir, save_weight=0, img_dim=img_dim)


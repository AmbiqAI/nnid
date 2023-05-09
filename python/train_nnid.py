"""
Training script for s2i RNN
"""
import os
import re
import logging
import argparse
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from nnsp_pack.nn_module_nnid import NeuralNetClass, lstm_states, tf_round
from nnsp_pack.tfrecord_converter_nnid import tfrecords_pipeline
from nnsp_pack.loss_functions import cross_entropy_nnid, contrast_nnid
from nnsp_pack.converter_fix_point import fakefix_tf
from nnsp_pack.calculate_feat_stats_nnid import feat_stats_estimator
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder
from nnsp_pack.nnid_correlation import gen_target_nnid, get_corr_fast
import c_code_table_converter
from data_nnid_ti import NUM_GROUP_PPLS, NUM_SENTS, NOISE_TYPES, SNR_DBS_MIN_MAX

physical_devices    = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: # pylint: disable=bare-except
    pass

SHOW_STEPS          = True
DISPLAY_HISTOGRAM   = True
TARGET = gen_target_nnid(NUM_GROUP_PPLS, NUM_SENTS)
EPS                 = 10**-5

# @tf.function
def train_kernel(
        feat, mask, target, states, net,
        optimizer,
        training    = True,
        quantized   = False):
    """
    Training kernel
    """
    with tf.GradientTape() as tape:
        est, states = net(
                feat, mask, states,
                training = training,
                quantized = quantized)
        est = est[:,-1,:] # take the last time step
        corr = get_corr_fast(
            est,
            target,
            NUM_GROUP_PPLS,
            NUM_SENTS,
            eps = EPS)

        corr = tf.math.abs(net.weight_cos + 10**-5) * corr + net.bias_cos

        if 1:
            est_target = tf.nn.softmax(corr, axis=-1)

            ave_loss, steps = cross_entropy_nnid(
                                target,
                                est_target)
        else:
            est_target = tf.sigmoid(corr)
            ave_loss, steps = contrast_nnid(
                            target,
                            est_target)

    if training:
        gradients = tape.gradient(ave_loss, net.trainable_variables)

        gradients_clips = [ tf.clip_by_norm(grad, 3) for grad in gradients ]
        optimizer.apply_gradients(zip(gradients_clips, net.trainable_variables))

    return est_target, states, ave_loss, steps

def epoch_proc( net,
                optimizer,
                dataset,
                fnames,
                batchsize,
                training,
                zero_state,
                norm_mean,
                norm_inv_std,
                num_context     = 6,
                quantized=False
                ):
    """
    Training for one epoch
    """
    total_batches = int(np.ceil(len(fnames) * len(fnames[0]) * len(fnames[0][1]) / batchsize))
    net.reset_stats()
    for batch, data in enumerate(dataset):
        tf.print(f"\r {batch}/{total_batches}: ",
                        end = '')

        feats, masks = data
        target = tf.identity(TARGET)
        batchsize0, _, dim_feat = feats.shape
        if 0:
            idx = 3
            feat = feats[idx,:,:].numpy()
            tar = target[idx,:].numpy() * 25
            plt.figure(1)
            plt.clf()
            plt.imshow(
                feat.T,
                origin      = 'lower',
                cmap        = 'pink_r',
                aspect      = 'auto')
            plt.plot(tar)
            plt.show()

        # initial input: 2^-15 in time domain
        shape = (batchsize0, num_context-1, dim_feat)
        padddings_tsteps_zeros = tf.constant(
                        np.full(shape, np.log10(2**-15)),
                        dtype = tf.float32)

        feats = tf.concat([padddings_tsteps_zeros, feats], 1)
        feats = (feats - norm_mean) * norm_inv_std
        feats = fakefix_tf(feats, 16, 8)

        states = lstm_states(net, batchsize, zero_state= zero_state)

        tmp = train_kernel(
                feats,
                masks,
                target,
                states,
                net,
                optimizer,
                training    = training,
                quantized   = quantized)

        est_target, states, ave_loss, steps = tmp
        est_target  = tf.math.argmax(est_target,axis=-1)
        target      = tf.math.argmax(target,    axis=-1)

        net.update_cost_steps(ave_loss, steps)

        net.update_accuracy(
            target,
            est_target)

        net.show_loss(
            net.stats['acc_loss'],
            net.stats['acc_matchCount'],
            net.stats['acc_steps'],
            SHOW_STEPS)

    tf.print('\n', end = '')

def main(args):
    """
    main function to train neural network training
    """
    batchsize       = NUM_GROUP_PPLS * NUM_SENTS
    num_epoch       = args.num_epoch
    epoch_loaded    = args.epoch_loaded
    quantized       = args.quantized

    tfrecord_list = {
        'train' : args.train_list,
        'test'  : args.test_list}

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    arch = load_nn_arch(args.nn_arch)
    neurons, drop_rates, layer_types, activations, num_context, num_dnsampl = arch

    folder_nn = setup_nn_folder(args.nn_arch)

    dim_feat = neurons[0]

    nn_train = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        dropRates   = drop_rates,
        activations = activations,
        batchsize   = batchsize,
        nDownSample = num_dnsampl,
        kernel_size = num_context)

    if epoch_loaded == 'random':
        epoch_loaded = -1
        loss = {'train' : np.zeros(num_epoch+1),
                'test' : np.zeros(num_epoch+1)}

        acc  = {'train' : np.zeros(num_epoch+1),
                'test' : np.zeros(num_epoch+1)}
        epoch1_loaded = epoch_loaded + 1
    else:
        if epoch_loaded == 'latest':
            checkpoint_dir = f'{folder_nn}/checkpoints'
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            nn_train.load_weights(latest)
            tmp = re.search(r'_ep(\d)+', latest)
            epoch_loaded = int(re.sub(r'_ep','',tmp.group(0)))
            epoch1_loaded = epoch_loaded + 1
        else:
            nn_train.load_weights(
                f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}')
            epoch1_loaded = epoch_loaded + 1

        print(f"Model at epoch {epoch1_loaded - 1} is retrieved")

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "rb") as file:
            loss = pickle.load(file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "rb") as file:
            acc = pickle.load(file)

        ax_handle = plt.subplot(2,1,1)
        ax_handle.plot(loss['train'][0: epoch_loaded+1])
        ax_handle.plot(loss['test'][0: epoch_loaded+1])
        ax_handle.legend(['train', 'test'])
        ax_handle.grid(True)
        ax_handle.set_title(f'Loss and accuracy upto epoch {epoch_loaded}. Close it to continue')

        ax_handle = plt.subplot(2,1,2)
        ax_handle.plot(acc['train'][0: epoch_loaded+1])
        ax_handle.plot(acc['test'][0: epoch_loaded+1])
        ax_handle.legend(['train', 'test'])
        ax_handle.set_xlabel('Epochs')
        print(f"(train) min loss epoch = {np.argmin(loss['train'][0:epoch_loaded+1])}")
        print(f"(test)  min loss epoch = {np.argmin(loss['test'][0:epoch_loaded+1])}")
        print(f"(train) max acc epoch = {np.argmax(acc['train'][0:epoch_loaded+1])}")
        print(f"(test)  max acc epoch = {np.argmax(acc['test'][0:epoch_loaded+1])}")

        ax_handle.grid(True)

        plt.show()

    fnames = {}
    for tr_set in ['train', 'test']:
        with open(tfrecord_list[tr_set], 'r') as file: # pylint: disable=unspecified-encoding
            fnames[tr_set] = yaml.load(file, Loader=SafeLoader)


    _, dataset = tfrecords_pipeline(
            fnames['train'],
            NUM_SENTS,
            len(NOISE_TYPES) * len(SNR_DBS_MIN_MAX),
            NUM_GROUP_PPLS,
            is_shuffle = True)

    _, dataset_tr = tfrecords_pipeline(
            fnames['train'],
            NUM_SENTS,
            len(NOISE_TYPES) * len(SNR_DBS_MIN_MAX),
            NUM_GROUP_PPLS,
            is_shuffle = False)

    _, dataset_te = tfrecords_pipeline(
            fnames['test'],
            NUM_SENTS,
            len(NOISE_TYPES) * len(SNR_DBS_MIN_MAX),
            NUM_GROUP_PPLS,
            is_shuffle=False)

    if os.path.exists(f'{folder_nn}/stats.pkl'):
        with open(os.path.join(folder_nn, 'stats.pkl'), "rb") as file:
            stats = pickle.load(file)
    else:
        stats = feat_stats_estimator(
                dataset_tr, fnames['train'],
                dim_feat, folder_nn,
                NUM_GROUP_PPLS,
                NUM_SENTS)

    nn_np = c_code_table_converter.tf2np(nn_train, quantized=quantized)
    if DISPLAY_HISTOGRAM:
        c_code_table_converter.draw_nn_hist(nn_np)
        c_code_table_converter.draw_nn_weight(
            nn_np,
            nn_train,
            pruning=False)
    for epoch in range(epoch1_loaded, num_epoch):
        t_start = tf.timestamp()
        tf.print(f'\n(EP {epoch})\n', end = '')

        # Training phase
        if 1:
            epoch_proc( nn_train,
                        optimizer,
                        dataset,
                        fnames['train'],
                        batchsize,
                        training        = True,
                        zero_state      = False,
                        norm_mean       = stats['nMean_feat'],
                        norm_inv_std    = stats['nInvStd'],
                        num_context     = num_context,
                        quantized       = quantized)

        # Computing Training loss
        epoch_proc( nn_train,
                    optimizer,
                    dataset_tr,
                    fnames['train'],
                    batchsize,
                    training        = False,
                    zero_state      = True,
                    norm_mean       = stats['nMean_feat'],
                    norm_inv_std    = stats['nInvStd'],
                    num_context     = num_context,
                    quantized       = quantized)

        loss['train'][epoch] = nn_train.stats['acc_loss'] / nn_train.stats['acc_steps']

        acc['train'][epoch] = nn_train.stats['acc_matchCount'] / nn_train.stats['acc_steps']

        # Computing Testing loss
        epoch_proc( nn_train,
                    optimizer,
                    dataset_te,
                    fnames['test'],
                    batchsize,
                    training            = False,
                    zero_state          = True,
                    norm_mean           = stats['nMean_feat'],
                    norm_inv_std        = stats['nInvStd'],
                    num_context         = num_context,
                    quantized           = quantized)

        loss['test'][epoch] = nn_train.stats['acc_loss'] / nn_train.stats['acc_steps']

        acc['test'][epoch] = nn_train.stats['acc_matchCount'] / nn_train.stats['acc_steps']

        nn_train.save_weights(f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch}')

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "wb") as file:
            pickle.dump(loss, file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "wb") as file:
            pickle.dump(acc, file)

        tf.print('Epoch spent ', tf_round(tf.timestamp() - t_start), ' seconds')
        print(f"(train) min loss epoch = {np.argmin(loss['train'][0:epoch+1])}")
        print(f"(test)  min loss epoch = {np.argmin(loss['test'][0:epoch+1])}")
        print(f"(train) max acc epoch = {np.argmax(acc['train'][0:epoch+1])}")
        print(f"(test)  max acc epoch = {np.argmax(acc['test'][0:epoch+1])}")

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Training script for NN-ID model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_id_nn_arch128x3_ti.txt',
        help='nn architecture')

    argparser.add_argument(
        '-st',
        '--train_list',
        default='data/train_tfrecords_nnid.yaml',
        help='train_list')

    argparser.add_argument(
        '-ss',
        '--test_list',
        default='data/test_tfrecords_nnid.yaml',
        help='test_list')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '-l',
        '--learning_rate',
        default =  4 * 10**-4,
        type=float,
        help='learning rate')

    argparser.add_argument(
        '-e',
        '--num_epoch',
        type=int,
        default=1000,
        help='Number of epochs to train')

    argparser.add_argument(
        '--epoch_loaded',
        default= "latest",
        help='epoch_loaded = \'random\': weight table is randomly generated, \
              epoch_loaded = \'latest\': weight table is loaded from the latest saved epoch result \
              epoch_loaded = 10  \
              (or any non-negative integer): weight table is loaded from epoch 10')

    main(argparser.parse_args())

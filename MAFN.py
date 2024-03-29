# BASED ON CODE FROM https://github.com/AudioVisualEmotionChallenge/AVEC2019/blob/master/Baseline_systems/CES/baseline_lstm.py
#

import os
import pandas as pd
import numpy as np
import keras.backend as K
from keras import models, layers
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, concatenate, Concatenate, TimeDistributed, Bidirectional, LSTM, Activation, Flatten, Lambda, Reshape, RepeatVector, Permute
from keras.optimizers import RMSprop
import h5py
import os
import sys

sys.path.append(os.getcwd()[:-12]+"Scripts")
from CES_data import load_CES_data
from calc_scores import calc_scores


from numpy.random import seed
from tensorflow import *
from Attention import ContextVector

if test.gpu_device_name():
    print('Default GPU Device: {}'.format(test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, dropout, num_targets):
    # Input layers
    ###################################################################################################################################
    audioEncoderInput = Input(shape=(max_seq_len[0],num_features[0]))
    decoderInput1 = Input(shape=(max_seq_len[0],1))
    decoderInput2 = Input(shape=(max_seq_len[0],1))
    decoderInput3 = Input(shape=(max_seq_len[0],1))
    decoderInput = concatenate([decoderInput1, decoderInput2, decoderInput3])
    
    audioAttention = Dense(1, activation='tanh')(audioEncoderInput)
    audioAttention = Flatten()(audioAttention)
    audioAttention = Activation('softmax')(audioAttention)
    audioAttention = RepeatVector(audioEncoderInput.shape[2])(audioAttention)
    audioAttention = Permute([2,1])(audioAttention)
    
    w_audio = layers.multiply([audioEncoderInput, audioAttention])
    
    audioEncoder = Bidirectional(LSTM(num_units_1, return_sequences=True, return_state=True))
    a_encoder_outs, a_fwd_h, a_fwd_c, a_bckwd_h, a_bckwd_c = audioEncoder(w_audio)
    a_state_h = Concatenate()([a_fwd_h, a_bckwd_h])
    a_state_c = Concatenate()([a_fwd_c, a_bckwd_c])    
    audioDecoder = LSTM(num_units_2, return_sequences=True, return_state=True)(decoderInput, initial_state=[a_state_h, a_state_c])
    ###################################################################################################################################
    ###################################################################################################################################
    visualEncoderInput = Input(shape=(max_seq_len[1],num_features[1]))
    
    
    visualAttention = Dense(1, activation='tanh')(visualEncoderInput)
    visualAttention = Flatten()(visualAttention)
    visualAttention = Activation('softmax')(visualAttention)
    visualAttention = RepeatVector(visualEncoderInput.shape[2])(visualAttention)
    visualAttention = Permute([2,1])(visualAttention)
    
    w_visual = layers.multiply([visualEncoderInput, visualAttention])
    
    visualEncoder = Bidirectional(LSTM(num_units_1, return_sequences=True, return_state=True))
    v_encoder_outs, v_fwd_h, v_fwd_c, v_bckwd_h, v_bckwd_c = visualEncoder(w_visual)
    v_state_h = Concatenate()([v_fwd_h, v_bckwd_h])
    v_state_c = Concatenate()([v_fwd_c, v_bckwd_c])    
    visualDecoder = LSTM(num_units_2, return_sequences=True, return_state=True)(decoderInput, initial_state=[v_state_h, v_state_c])
    ###################################################################################################################################

    attention = ContextVector()([concatenate([a_encoder_outs, v_encoder_outs]), concatenate([audioDecoder[0], visualDecoder[0]])])
    weighted_out = Concatenate(axis=-1)([concatenate([audioDecoder[0], visualDecoder[0]]), attention[0]])
    weighted_out = Dense(3, activation='softmax')(weighted_out)

    # Output layer (linear activation)
    outputs = []
    out1 = TimeDistributed(Dense(1))(weighted_out)
    outputs.append(out1)
    if num_targets>=2:
        out2 = TimeDistributed(Dense(1))(weighted_out)
        outputs.append(out2)
    if num_targets==3:
        out3 = TimeDistributed(Dense(1))(weighted_out)
        outputs.append(out3)
    
    # Create and compile model
    rmsprop = RMSprop(lr=learning_rate)
    model   = Model(inputs=[audioEncoderInput, decoderInput1, decoderInput2, decoderInput3, visualEncoderInput], outputs=outputs)
    model.compile(optimizer=rmsprop, loss=ccc_loss)  # CCC-based loss function
    return model

def main(audio_features_folders=['Audio/audio_features_egemaps_xbow/'], visual_features_folders=['Visual/visual_features_xbow/'], path_output='predictions/', audioFileEndings=['.csv'], visualFileEndings=['.csv']):

    # Input
    #features_folders = ['audio_features_egemaps_xbow/', 'visual_features_xbow/']  # Select features to be considered (must have the same hop size as the labels, i.e., 0.1s for CES)
    #path_output = 'predictions/'  # To store the predictions on the test (& development) partitions
    
    ## Configuration
    base_folder      = '../../'  # features and train/development labels
    targets          = ['arousal','valence','liking']  # Targets to be learned at the same time
    
    output_predictions_devel = True  # Write predictions on development set
    test_available           = False  # True, if test features are available
    
    # Neural net parameters
    standardise   = True     # Standardise the input features (0 mean, unit variance)
    batch_size    = 1        # Full-batch: 68 sequences
    learning_rate = 0.001    # default is 0.001
    num_epochs    = 100       # Number of epochs
    num_units_1   = 32       # Number of LSTM units in LSTM layer 2
    num_units_2   = 64       # Number of LSTM units in LSTM layer 2
    bidirectional = False    # True/False
    dropout       = 0.1      # Dropout
    
    # Labels
    shift_sec     = 2.0      # Shift of annotations for training (in seconds)
    ## End Configuration
    
    ## Training
    labels_per_sec = 10  # 100ms hop size
    shift          = int(np.round(shift_sec*labels_per_sec))
    num_targets    = len(targets)
    
    # Set seeds to make results reproducible 
    # (Note: Results might be different from those reported by the Organisers as training of the seeds depend on hardware!
    seed(1)
    random.set_seed(2)

    # Load AVEC2019_CES data
    print('Loading data ...')
    audio_train_DE_x, audio_train_HU_x, train_DE_y, train_HU_y, audio_devel_DE_x, audio_devel_HU_x, _, _, _, _, _ = load_CES_data(base_folder, audio_features_folders, targets, test_available, audioFileEndings)
    visual_train_DE_x, visual_train_HU_x, _, _, visual_devel_DE_x, visual_devel_HU_x, devel_DE_labels_original, devel_HU_labels_original, _, _, _ = load_CES_data(base_folder, visual_features_folders, targets, test_available, visualFileEndings)

    # Concatenate German and Hungarian cultures for Training and Development
    audio_train_x = np.concatenate((audio_train_DE_x, audio_train_HU_x), axis=0)
    visual_train_x = np.concatenate((visual_train_DE_x, visual_train_HU_x), axis=0)

    audio_devel_x = np.concatenate((audio_devel_DE_x, audio_devel_HU_x), axis=0)
    visual_devel_x = np.concatenate((visual_devel_DE_x, visual_devel_HU_x), axis=0)

    train_y = []
    devel_labels_original = []
    for t in range(0,num_targets):
        train_y.append( np.concatenate((train_DE_y[t], train_HU_y[t]), axis=0) )
        devel_labels_original.append( devel_DE_labels_original[t] + devel_HU_labels_original[t] )
    
    # Get some stats
    max_seq_len  = (audio_train_x.shape[1], visual_train_x.shape[1])   # same for all partitions
    num_features = (audio_train_x.shape[2], visual_train_x.shape[2])
    print(' ... done')
    
    if standardise:
        MEAN_audio, STDDEV_audio = standardise_estimate(audio_train_x)
        MEAN_visual, STDDEV_visual = standardise_estimate(visual_train_x)
        standardise_apply(audio_train_x, MEAN_audio, STDDEV_audio)
        standardise_apply(audio_devel_x, MEAN_audio, STDDEV_audio)
        standardise_apply(audio_devel_DE_x, MEAN_audio, STDDEV_audio)
        standardise_apply(audio_devel_HU_x, MEAN_audio, STDDEV_audio)

        standardise_apply(visual_train_x, MEAN_visual, STDDEV_visual)
        standardise_apply(visual_devel_x, MEAN_visual, STDDEV_visual)
        standardise_apply(visual_devel_DE_x, MEAN_visual, STDDEV_visual)
        standardise_apply(visual_devel_HU_x, MEAN_visual, STDDEV_visual)

      # standardise_apply(test_DE_x, MEAN, STDDEV)
      # standardise_apply(test_HU_x, MEAN, STDDEV)
      # standardise_apply(test_CN_x, MEAN, STDDEV)
    # Shift labels to compensate annotation delay
    print('Shifting training labels to the front for ' + str(shift_sec) + ' seconds ...')
    for t in range(0, num_targets):
        train_y[t] = shift_labels_to_front(train_y[t], shift)
    print(' ... done')


    decoder_train_y = np.roll(train_y, 1, axis=-2)
    for i in range(3):
        for j in range(len(decoder_train_y[i])):
            decoder_train_y[i][j][0][0] = 0

    decoder_devel_labels_og = list(np.roll(devel_labels_original, 1, axis=-2))
    decoder_devel_labels_original = [[], [], []]
    for i in range(3):
        for j in range(len(decoder_devel_labels_og[i])):
            decoder_devel_labels_og[i][j][0][0] = 0
            frames=[]
            for k in range(len(decoder_devel_labels_og[i][j])):
                frames.append([decoder_devel_labels_og[i][j][k][0]])
            for k in range(max(max_seq_len[0], max_seq_len[1])-len(decoder_devel_labels_og[i][j])):
                frames.append([0])
            decoder_devel_labels_original[i].append(frames)
    decoder_devel_labels_original = np.array(decoder_devel_labels_original)        
            
    
    decoder_devel_DE_labels_og = list(np.roll(devel_DE_labels_original, 1, axis=-2))
    decoder_devel_DE_labels_original = [[], [], []]
    for i in range(3):
        for j in range(len(decoder_devel_DE_labels_og[i])):
            decoder_devel_DE_labels_og[i][j][0][0] = 0
            frames=[]
            for k in range(len(decoder_devel_DE_labels_og[i][j])):
                frames.append([decoder_devel_DE_labels_og[i][j][k][0]])
            for k in range(max(max_seq_len[0], max_seq_len[1])-len(decoder_devel_DE_labels_og[i][j])):
                frames.append([0])
            decoder_devel_DE_labels_original[i].append(frames)

    decoder_devel_DE_labels_original = np.array(decoder_devel_DE_labels_original)  

    
    decoder_devel_HU_labels_og = list(np.roll(devel_HU_labels_original, 1, axis=-2))
    decoder_devel_HU_labels_original = [[], [], []]
    for i in range(3):
        for j in range(len(decoder_devel_HU_labels_og[i])):
            decoder_devel_HU_labels_og[i][j][0][0] = 0
            frames=[]
            for k in range(len(decoder_devel_HU_labels_og[i][j])):
                frames.append([decoder_devel_HU_labels_og[i][j][k][0]])
            for k in range(max(max_seq_len[0], max_seq_len[1])-len(decoder_devel_HU_labels_og[i][j])):
                frames.append([0])
            decoder_devel_HU_labels_original[i].append(frames)

    decoder_devel_HU_labels_original = np.array(decoder_devel_HU_labels_original) 

            
    # Create model
    model = emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, dropout, num_targets)
    print(model.summary())
    
    # Structures to store results (development)
    ccc_devel_best = np.ones(num_targets) * -1.
    
    # Train and evaluate model
    epoch = 1
    while epoch <= num_epochs:
        model.fit([audio_train_x, decoder_train_y[0], decoder_train_y[1], decoder_train_y[2], visual_train_x], train_y, batch_size=batch_size, initial_epoch=epoch-1, epochs=epoch)  # Evaluate after each epoch
        
        # Evaluate on development partition
        ccc_iter = evaluate_devel(model, [audio_devel_x, decoder_devel_labels_original[0], decoder_devel_labels_original[1], decoder_devel_labels_original[2], visual_devel_x], devel_labels_original, shift, targets)
        
        # Print results
        print('CCC Development (' + ','.join(targets) + '): ' + str(np.round(ccc_iter*1000)/1000))
        
        # If CCC on the development partition improved, store the best models as HDF5 files
        for t in range(0, num_targets):
            if ccc_iter[t] > ccc_devel_best[t]:
                ccc_devel_best[t] = ccc_iter[t]
                save_model( model, targets[t]+'.hdf5' )
                """
                file1 = h5py.File(targets[t]+'.hdf5', 'w')
                weight = model.get_weights()
                for i in range(len(weight)):
                    file1.create_dataset('weight'+str(i),data=weight[i])
                file1.close()
                """
                
        # Next epoch
        epoch += 1
    # ... Training finished
    # Print best results on development partition
    print('CCC Development best (' + ','.join(targets) + '): ' + str(np.round(ccc_devel_best*1000)/1000))
    
    
    if output_predictions_devel:
        print('Getting predictions on Devel and shifting back')
        pred_devel_DE = []
        pred_devel_HU = []
        for t in range(0, num_targets):
            if num_targets==1:
                pred_devel_DE.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False, custom_objects={"ContextVector": ContextVector}).predict([audio_devel_DE_x, decoder_devel_DE_labels_original[0], decoder_devel_DE_labels_original[1], decoder_devel_DE_labels_original[2], visual_devel_DE_x]), shift) )
                pred_devel_HU.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False, custom_objects={"ContextVector": ContextVector}).predict([audio_devel_HU_x, decoder_devel_HU_labels_original[0], decoder_devel_HU_labels_original[1], decoder_devel_HU_labels_original[2], visual_devel_HU_x]), shift) )
            else:
                pred_devel_DE.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False, custom_objects={"ContextVector": ContextVector}).predict([audio_devel_DE_x, decoder_devel_DE_labels_original[0], decoder_devel_DE_labels_original[1], decoder_devel_DE_labels_original[2], visual_devel_DE_x])[t], shift) )
                pred_devel_HU.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False, custom_objects={"ContextVector": ContextVector}).predict([audio_devel_HU_x, decoder_devel_HU_labels_original[0], decoder_devel_HU_labels_original[1], decoder_devel_HU_labels_original[2], visual_devel_HU_x])[t], shift) )
        print('Writing predictions on Devel partitions for the best models (best CCC on the Development partition) for each dimension into folder ' + path_output)
        write_predictions(path_output, pred_devel_DE, targets, prefix='Devel_DE_', labels_per_sec=labels_per_sec)
        write_predictions(path_output, pred_devel_HU, targets, prefix='Devel_HU_', labels_per_sec=labels_per_sec)
    
    # Get predictions on test (and shift back) Write best predictions
    if test_available:
        print('Getting predictions on Test and shifting back')
        pred_test_DE = []
        pred_test_HU = []
        pred_test_CN = []
        for t in range(0, num_targets):
            if num_targets==1:
                pred_test_DE.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_DE_x), shift) )
                pred_test_HU.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_HU_x), shift) )
                pred_test_CN.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_CN_x), shift) )
            else:
                pred_test_DE.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_DE_x)[t], shift) )
                pred_test_HU.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_HU_x)[t], shift) )
                pred_test_CN.append( shift_labels_to_back( load_model(targets[t]+'.hdf5',compile=False).predict(test_CN_x)[t], shift) )
        print('Writing predictions on Test partitions for the best models (best CCC on the Development partition) for each dimension into folder ' + path_output)
        write_predictions(path_output, pred_test_DE, targets, prefix='Test_DE_', labels_per_sec=labels_per_sec)
        write_predictions(path_output, pred_test_HU, targets, prefix='Test_HU_', labels_per_sec=labels_per_sec)
        write_predictions(path_output, pred_test_CN, targets, prefix='Test_CN_', labels_per_sec=labels_per_sec)



def standardise_estimate(train):
    # Estimate parameters (masked parts are not considered)
    num_features  = train.shape[2]
    estim         = train.reshape([-1,num_features])
    estim_max_abs = np.max(np.abs(estim), axis=1)
    mask          = np.where(estim_max_abs>0)[0]
    MEAN          = np.mean(estim[mask], axis=0)
    STDDEV        = np.std(estim[mask], axis=0)
    return MEAN, STDDEV

def standardise_apply(partition, MEAN, STDDEV):
    # Standardise partition with given parameters
    for sub in range(0, partition.shape[0]):
        part_max_abs = np.max(np.abs(partition[sub,:,:]), axis=1)
        mask         = np.where(part_max_abs>0)[0]        
        partition[sub,mask,:] = partition[sub,mask,:] - MEAN
        partition[sub,mask,:] = partition[sub,mask,:] / (STDDEV + np.finfo(np.float32).eps)
    return partition  # not required


def evaluate_devel(model, devel_x , label_devel, shift, targets):
    # Evaluate performance (CCC) on the development set
    #  -shift back the predictions in time
    #  -use the original labels (without zero padding)
    num_targets = len(targets)
    CCC_devel   = np.zeros(num_targets)
    # Get predictions
    pred_devel = model.predict(devel_x)
    # In case of a single target, model.predict() does not return a list, which is required
    if num_targets==1:
        pred_devel = [pred_devel]    
    for t in range(0,num_targets):
        # Shift predictions back in time (delay)
        pred_devel[t] = shift_labels_to_back(pred_devel[t], shift)
        CCC_devel[t]  = evaluate_partition(pred_devel[t], label_devel[t])
    return CCC_devel


def evaluate_partition(pred, gold):
    # pred: np.array (num_seq, max_seq_len, 1)
    # gold: list (num_seq) - np.arrays (len_original, 1)
    pred_all = np.array([])
    gold_all = np.array([])
    for n in range(0, len(gold)):
        # cropping to length of original sequence
        len_original = len(gold[n])
        pred_n = pred[n,:len_original,0]
        # global concatenation - evaluation
        pred_all = np.append(pred_all, pred_n.flatten())
        gold_all = np.append(gold_all, gold[n].flatten())
    ccc, _, _ = calc_scores(gold_all,pred_all)
    return ccc


def write_predictions(path_output, predictions, targets, prefix='Test_DE_', labels_per_sec=10):
    num_targets = len(targets)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    for n in range(0, predictions[0].shape[0]):
        seq_len   = predictions[0].shape[1]
        pred_inst = np.empty((seq_len,num_targets))
        for t in range(0,num_targets):
            pred_inst[:,t] = predictions[t][n,:,0]
        # add time stamp
        time_stamp = np.linspace(0., (seq_len-1)/float(labels_per_sec), seq_len).reshape(-1,1)
        pred_inst  = np.concatenate( (time_stamp, pred_inst), axis=1 )
        # create data frame and write file
        instname = prefix + str(int(n+1)).zfill(2)
        filename = path_output + instname + '.csv'
        data_frame         = pd.DataFrame(pred_inst, columns=['timestamp']+targets)
        data_frame['name'] = '\'' + instname + '\''
        data_frame.to_csv(filename, sep=';', columns=['name','timestamp']+targets, index=False, float_format='%.6f')


def shift_labels_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels


def shift_labels_to_back(labels, shift=0):
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels


def ccc_loss(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    ccc_loss   = K.constant(1.) - ccc
    return ccc_loss


if __name__ == '__main__':
    main(audio_features_folders=['Audio/audio_features_egemaps/', 'Audio/audio_features_mfcc/'], 
            visual_features_folders=['Visual/visual_features/'], 
            path_output= 'AMAFN/predictions/', audioFileEndings=['.csv', '.csv'], visualFileEndings=['.csv'])



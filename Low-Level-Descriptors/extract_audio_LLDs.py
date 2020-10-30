#!/bin/python2
# python2 script
# Extract acoustic LLDs (MFCC and eGeMAPS sets from openSMILE)
# Output: csv files

import os
import time

def extract_audioLLDs(feature_type, input_file):
    # MODIFY HERE
    # feature_type is 'mfcc' or 'egemaps'        
    #folder_data   = 'audio/'  # folder with audio (.wav) files
    exe_opensmile = 'opensmile-2.3.0/SMILExtract'  # MODIFY this path to the folder of the SMILExtract (version 2.3) executable
    path_config   = 'opensmile-2.3.0/config/'            # MODIFY this path to the config folder of opensmile 2.3 - no POSIX here on cygwin (windows)
    
    if feature_type=='mfcc':
        folder_output = 'audio_features/audio_features_mfcc/'  # output folder
        conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
        opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
        outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
    elif feature_type=='egemaps':
        folder_output = 'audio_features/audio_features_egemaps/'  # output folder
        conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
        opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
        outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc
    else:
        print('Error: Feature type ' + feature_type + ' unknown!')

    if not os.path.exists('audio_features'):
        os.mkdir('audio_features')
    
    if not os.path.exists(folder_output):
        os.mkdir(folder_output)

    instname    = input_file.split("/")[-1].split('.')[0]
    outfilename = folder_output + instname + '.csv'
    opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + input_file + ' ' + outputoption + ' ' + outfilename + ' -instname ' + instname + ' -output ?'  # (disabling htk output)
    
    print(opensmile_call)
    os.system(opensmile_call)
    time.sleep(0.01)

    #os.remove('smile.log')
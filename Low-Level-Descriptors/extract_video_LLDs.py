#!/bin/python2
# python2 script
# Extract visual LLDs (FAU likelihoods) for video files of AVEC 2019
# Output: csv files
import sys
import os
import time
import numpy as np
from read_csv import load_features
from write_csv import save_features
import subprocess
import cv2

def extract_videoLLDs(input_file):
  
    
    folder_output = 'visual_features_LLD/'  # output folder
    exe_openface  = 'OpenFace-master/build/bin/FeatureExtraction'  # MODIFY this path to the folder of OpenFace (version 1.0.0): https://github.com/TadasBaltrusaitis/OpenFace

    conf_openface = '-aus'  # Facial Action Units

    if not os.path.exists(folder_output):
        os.mkdir(folder_output)

    # Header for visual feature files (FAUs)
    header_output_file = 'name;frameTime;confidence;AU01_r;AU02_r;AU04_r;AU05_r;AU06_r;AU07_r;AU09_r;AU10_r;AU12_r;AU14_r;AU15_r;AU17_r;AU20_r;AU23_r;AU25_r;AU26_r;AU45_r'  # 17 AU intensities

        
    instname      = input_file.split("/")[-1].split('.')[0]
    outfilename   = folder_output + instname + '.csv'

    openface_call = exe_openface + ' ' + conf_openface + ' -f ' + input_file + ' -out_dir ' + folder_output
    #with open("openface_call.sh", "w") as f:
    #    f.write("#!/bin/bash\n"+openface_call)

    #subprocess.check_output(['./openface_call.sh'], shell=True)
        
    os.system(openface_call)
    time.sleep(0.01)
    
    # Re-format files (as required by, e.g., openXBOW)
    features = load_features(outfilename, skip_header=True, skip_instname=False, delim=',')
    features = np.delete(features, [0,1,4,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], 1)  # removing: frame, face_id, confidence, FAU present (c, 1/0)
    save_features(outfilename, features, append=False, instname=instname, header=header_output_file, delim=';', precision=3)
    
    # Remove details file
    #os.remove(folder_output + instname + '_of_details.txt')



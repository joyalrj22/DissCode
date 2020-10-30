ACKNOWLEDGEMENTS:
All the code in Low-Level-Descriptors/, OpenFace-master/build/bin/demo.py, OpenFace-master/build/bin/feature_extraction/ is from the AVEC baseline GitHub repository: https://github.com/AudioVisualEmotionChallenge/AVEC2019/tree/master/Baseline_features_extraction

Description of key files:

EarlyAttentionFusion.py: 
	This is our Early Attention Fusion Model. The code for the architecture is our own, the code for training and running etc. is written using https://github.com/AudioVisualEmotionChallenge/AVEC2019/blob/master/Baseline_systems/CES/baseline_lstm.py (AVEC Baseline code) as a starting point

Encoder-DecoderModel.py:
	The Attention based encoder-decoder model. Training code based upon AVEC Baseline code

MAFN.py:
	The Multi-Attention Fusion Network model. Training code based upon AVEC Baseline code

Attention.py:
	Attention Layer for the encoder/decoder model

frontend.py:
	Classes for audio and video recording ability of GUI
GUI.py:
	Software demonstrator


TO RUN THE DEMONSTRATOR:
Download and install OpenFace from https://github.com/TadasBaltrusaitis/OpenFace and name the folder OpenFace-master. 
Activate virtual environment with source ds_virtualenv/bin/activate. Not that the program only works on linux. Not all dependencies could be installed into the virtual environment so make sure ffmpeg support is included in linux with all basic video and audio codecs installed. Also make sure port audio is installed from: http://www.portaudio.com/

To run: python3.6 GUI.py




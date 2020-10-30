from tkinter import *
import tkinter.filedialog as filedialog
#from tkinter.ttk import *  
from tkinter import Label
import PIL.Image, PIL.ImageTk
import cv2
import threading
import subprocess
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from frontend import VideoRecorder, AudioRecorder
sys.path.append(os.getcwd()+"/Low-Level-Descriptors/")
from extract_video_LLDs import extract_videoLLDs
from extract_audio_LLDs import extract_audioLLDs
sys.path.append(os.getcwd()+"/OpenFace-master/build/bin/")
from demo import extract_deep_visual
import time
from load_features import load_llds, summarise_audio, load_deep_feature
from attention import ContextVector
from emotion_model import emotion_model
import h5py
import random

from keras.models import load_model
import keras.backend as K


#TODO: Plug in model 

#TODO: playing while seeking bug
#TODO: Aesthetics 

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


class Application:
    def __init__(self, master, window_title):
        self.master = master
        self.master.title(window_title)
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        self,master.configure(bg='white')
        
        self.playback_mode = False

        self.file_name = "example_1"
        
        self.mouse_move_handler = None
        self.infocus_axes = None

        self.arousal_bounds = (None, None)
        self.valence_bounds = (None, None)
        self.liking_bounds = (None, None)
        self.current_frame_img =None
        
        
        self.vid_path="video/example_1.avi"
        self.aud_path="audio/example_1.wav"
        self.paused=True
        self.render_widgets()
        self.master.mainloop()

    def store_bounds(self):
        self.arousal_bounds = (self.arousal_graph.get_xlim(), self.arousal_graph.get_ylim())
        self.valence_bounds = (self.valence_graph.get_xlim(), self.valence_graph.get_ylim())
        self.liking_bounds = (self.liking_graph.get_xlim(), self.liking_graph.get_ylim())

    def set_bounds(self):
        self.arousal_graph.set_xlim(self.arousal_bounds[0]), self.arousal_graph.set_ylim(self.arousal_bounds[1])
        self.valence_graph.set_xlim(self.valence_bounds[0]), self.valence_graph.set_ylim(self.valence_bounds[1])
        self.liking_graph.set_xlim(self.liking_bounds[0]), self.liking_graph.set_ylim(self.liking_bounds[1])

    def render_widgets(self):
        
        self.start_audiovisual_feed()
        
        self.graph_frame = Frame(self.master, width=642, height=482, bg='white')
        self.graph_frame.grid(row=0, column=1, rowspan=7)

        self.video_controls = Frame(self.master, width=self.video_source.sizex, bg='white')
        self.video_controls.grid(row=7, column=0, sticky=(N, W, E))

        self.recordButton = Button(self.video_controls, text= "Record", command=self.record, bg='light sky blue')
        self.recordButton.grid(row=0, column=0, padx=(5, 2))

        self.loadAudio = Button(self.video_controls,text='Load Audio File', command=self.load_audio_file, bg='light sky blue')
        self.loadAudio.grid(row=2, column=0, padx=(5,2))

        self.loadVideo = Button(self.video_controls,text='Load Video File', command=self.load_video_file, bg='light sky blue')
        self.loadVideo.grid(row=2, column=1)

        self.load_custom = Button(self.video_controls, text='Analyse Loaded Files', command=self.extractFeatures, bg='light sky blue')
        self.load_custom['state'] = 'disabled'
        self.load_custom.grid(row=2, column=2)

        self.options_frame = Frame(self.master, bg='white')
        self.options_frame.grid(row=8, column=0, rowspan=2, columnspan=2)

    def load_video_file(self):
        self.vid_path = filedialog.askopenfilename()
        if self.vid_path:
            self.file_name = self.vid_path.split('/')[-1].split('.')[0]
            if self.aud_path != "audio/example_1.wav" and self.aud_path:
                self.load_custom['state'] =  'normal'
            else:
                self.load_custom['state'] = "disabled"

    def load_audio_file(self):
        self.aud_path = filedialog.askopenfilename()
        if self.aud_path:
            self.file_name = self.aud_path.split('/')[-1].split('.')[0]

            if self.vid_path != "video/example_1.avi" and self.vid_path:
                self.load_custom['state'] =  'normal'
            else:
                self.load_custom['state'] = "disabled"
        
    def mouse_move_event(self, event):
        self.store_bounds()

        self.arousal_graph.cla(), self.valence_graph.cla(), self.liking_graph.cla()
        self.arousal_graph.title.set_text('Arousal')
        self.valence_graph.title.set_text('Valence')
        self.liking_graph.title.set_text('Liking')
        self.arousal_graph.plot(self.arousal_values)
        self.valence_graph.plot(self.valence_values)
        self.liking_graph.plot(self.liking_values)
        
        arousal_y = np.interp(event.xdata,range(len(self.arousal_values[0])),self.arousal_values[0])
        valence_y = np.interp(event.xdata,range(len(self.valence_values[0])),self.valence_values[0])
        liking_y = np.interp(event.xdata,range(len(self.liking_values[0])),self.liking_values[0])
        
        self.toolbar.set_message("Frame: {} \nArousal: {}  |  Valence: {}  |  Liking: {}".format(round(event.xdata, 2), round(arousal_y, 2), round(valence_y,2), round(liking_y, 2)))
        
        self.set_bounds()

        if self.infocus_axes is self.arousal_graph:
            if (event.xdata > self.plots[0][0].get_data()[0][-1]) or (event.xdata < self.plots[0][0].get_data()[0][0]):
                self.toolbar.set_message("")
                self.canvas.draw()
                return
            self.arousal_graph.scatter(event.xdata, arousal_y, s=50, c='red')
            self.valence_graph.scatter(event.xdata, valence_y, s=50, c='blue')
            self.liking_graph.scatter(event.xdata, liking_y, s=50, c='blue')

        elif self.infocus_axes is self.valence_graph:
            if (event.xdata > self.plots[1][0].get_data()[0][-1]) or (event.xdata < self.plots[1][0].get_data()[0][0]):
                self.toolbar.set_message("")
                self.canvas.draw()
                return
            self.arousal_graph.scatter(event.xdata, arousal_y, s=50, c='blue')
            self.valence_graph.scatter(event.xdata, valence_y, s=50, c='red')
            self.liking_graph.scatter(event.xdata, liking_y, s=50, c='blue')

        elif self.infocus_axes is self.liking_graph:
            if (event.xdata > self.plots[2][0].get_data()[0][-1]) or (event.xdata < self.plots[2][0].get_data()[0][0]):
                self.toolbar.set_message("")
                self.canvas.draw()
                return
            self.arousal_graph.scatter(event.xdata, arousal_y, s=50, c='blue')
            self.valence_graph.scatter(event.xdata, valence_y, s=50, c='blue')
            self.liking_graph.scatter(event.xdata, liking_y, s=50, c='red')
        else:
            self.arousal_graph.scatter(event.xdata, arousal_y, s=50, c='blue')
            self.valence_graph.scatter(event.xdata, valence_y, s=50, c='blue')
            self.liking_graph.scatter(event.xdata, liking_y, s=50, c='blue')

        self.canvas.draw()
    
    def axes_enter_event(self, event):
        self.mouse_move_handler = self.canvas.mpl_connect('motion_notify_event', self.mouse_move_event)
        self.infocus_axes = event.inaxes

    def axes_leave_event(self, event):
        self.canvas.mpl_disconnect(self.mouse_move_handler)
        self.mouse_move_handler = None
        self.infocus_axes = None
        self.arousal_graph.cla(), self.valence_graph.cla(), self.liking_graph.cla()
        self.arousal_graph.title.set_text('Arousal')
        self.valence_graph.title.set_text('Valence')
        self.liking_graph.title.set_text('Liking')
        self.set_bounds()
        self.arousal_graph.plot(self.arousal_values)
        self.valence_graph.plot(self.valence_values)
        self.liking_graph.plot(self.liking_values)
        self.canvas.draw()

    def plotChart(self):
        #self.arousal_values = pd.DataFrame([1,1,2,3,3,5,2])
        
        #self.valence_values = pd.DataFrame([0,1,22,10,5,4,2])
        
        #self.liking_values = pd.DataFrame([0.1,0.1,0.4,0.8,0.7,0.2,0.3])
        
        self.plots = []
        fig = Figure()
        self.arousal_graph = fig.add_subplot(311)
        self.arousal_graph.title.set_text('Arousal')
        self.plots.append(self.arousal_graph.plot(self.arousal_values))

        
        self.valence_graph = fig.add_subplot(312)
        self.valence_graph.title.set_text('Valence')
        self.plots.append(self.valence_graph.plot(self.valence_values))
        self.liking_graph = fig.add_subplot(313)
        self.liking_graph.title.set_text('Liking')
        self.plots.append(self.liking_graph.plot(self.liking_values))

        fig.subplots_adjust(hspace=0.6)

        self.canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        
        self.canvas.mpl_connect('axes_enter_event', self.axes_enter_event)
        self.canvas.mpl_connect('axes_leave_event', self.axes_leave_event)
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack()
 
    def start_audiovisual_feed(self): 
        self.video_source = VideoRecorder(name=self.vid_path)
        self.video_source.start()
        self.audio_source = AudioRecorder(filename=self.aud_path)
        
        self.video_frame = Canvas(self.master, width=self.video_source.sizex, height=self.video_source.sizey)
        self.feed_dims=(self.video_source.sizex, self.video_source.sizey)
        self.video_frame.grid(row=0, column=0, rowspan=7)

        self.delay = 15
        self.update()

    def move_x_Frames(self, x=1):
        self.frame_num.set(self.frame_num.get()+x)
        self.seek()

    def enable_playback_mode(self):
        self.playback_mode = True
        self.recordButton["text"] = "Record Mode"
        self.plotChart()
        
        
        self.video_source = cv2.VideoCapture(self.vid_path)
        
        self.max_frame_num = int(self.video_source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = IntVar()
        self.frame_num.trace("w", lambda x,y,z: self.mouse_move_event(type('obj', (object,), {'xdata': self.frame_num.get()})))
        self.frame_num.set(0)
        
        self.paused = True
        self.play_button = Button(self.video_controls, text="Play", command=self.toggle_play, bg='light sky blue')
        self.play_button.grid(row=1, column=0) 

        self.back_1_frame = Button(self.video_controls, text="<<", command=lambda: self.move_x_Frames(-1), width=1, bg='light sky blue')
        self.back_1_frame.grid(row=1, column=1)

        self.skip_1_frame = Button(self.video_controls, text=">>", command=self.move_x_Frames, width=1, bg='light sky blue')
        self.skip_1_frame.grid(row=1, column=2)

        

        self.seek_bar = Scale(self.video_controls, orient=HORIZONTAL ,from_=0, to_=self.max_frame_num, variable=self.frame_num, command=self.seek, length=350, fg='light sky blue', bg='white')
        self.seek_bar.grid(row=1, column=3, sticky=(E, W))

        
        self.audio_source = None  

    def toggle_play(self):
        self.paused = not self.paused
        if self.paused:
            self.play_button["text"] = "Play"
            return
        self.play_button["text"] = "Pause"

    def seek(self, *args):
        already_paused = self.paused
        self.paused = True
        
        self.video_source.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num.get())

        if already_paused:
            ret, frame = self.video_source.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                resized_img = PIL.Image.fromarray(frame).resize((1300, 1000), PIL.Image.ANTIALIAS)
                self.photo = PIL.ImageTk.PhotoImage(image=resized_img)
                if self.current_frame_img:
                    self.video_frame.itemconfig(self.current_frame_img, image=self.photo)
                else:
                    self.video_frame.create_image(self.feed_dims[0]//2, self.feed_dims[1]//2, image=self.photo)

        if not already_paused:
            self.paused = False

    def update(self):
        if self.playback_mode:
            ret = False
            if not self.paused:
                ret, frame = self.video_source.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frame_num.set(self.frame_num.get()+1)
        else:
            ret, frame = self.video_source.current_frame

        if ret:
            resized_img = PIL.Image.fromarray(frame).resize((1300, 1000), PIL.Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=resized_img)
            if self.current_frame_img:
                self.video_frame.itemconfig(self.current_frame_img, image=self.photo)
            else:
                self.video_frame.create_image(self.feed_dims[0]//2, self.feed_dims[1]//2, image=self.photo)

        self.master.after(self.delay, self.update)
            
    def record(self):
        if self.playback_mode:
            self.rnn_thread = None
            self.playback_mode = False
            self.file_name="example_1"
            self.vid_path="video/example_1.avi"
            self.aud_path="audio/example_1.wav"
            self.play_button.grid_forget()
            self.back_1_frame.grid_forget()
            self.seek_bar.grid_forget()
            self.skip_1_frame.grid_forget()
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.destroy()
            self.video_source.release()
            cv2.destroyAllWindows()
            self.video_source = VideoRecorder(name=self.file_name+".avi")
            self.video_source.start()
            self.audio_source = AudioRecorder(filename=self.file_name+".wav")
            self.recordButton["text"] = "Record"
            self.loadAudio['state'] = 'normal'
            self.loadVideo['state'] = 'normal'
            return

        self.audio_source.start()
        self.video_source.start(mode="record")

        self.recordButton["text"] = "Stop Recording"
        self.recordButton["command"] = self.stop_recording

    def stop_recording(self):
        self.recordButton["text"] = "Record"
        self.recordButton["command"] = self.record
        self.extractFeatures()

    def close(self):
        if self.playback_mode and self.video_source.isOpened():
            self.video_source.release()
        elif not self.playback_mode:
            self.video_source.delete()

        self.master.destroy()

    def loading_animation(self, ind):
        
        if not self.vid_extract_thread.isAlive() and not self.aud_extract_thread.isAlive():
            print("Extracting emotion values ...")
            self.extract_emotion_values()
            self.loading_label.pack_forget()
        
            self.recordButton['state'] = 'normal'
            self.enable_playback_mode()
            return

        if ind>39: ind = 0
        self.loading_label.configure(image=self.loading_frames[ind])
        ind += 1
        self.graph_frame.after(25, self.loading_animation, ind)

    def extractFeatures(self):
        self.load_custom['state'] = 'disabled'
        self.loadAudio['state'] = 'disabled'
        self.loadVideo['state'] = 'disabled'
        self.recordButton['state'] = 'disabled'
        self.loading_label = Label(self.graph_frame, width=642, height=482)
        self.loading_label.pack()
        self.loading_frames = [PhotoImage(file='./loading.gif', format='gif -index %i' %(i)) for i in range(40)]
         
        def extract_video_features():
            extract_videoLLDs(self.vid_path)
    
            
        def extract_audio_features():
            extract_audioLLDs('egemaps',self.aud_path)
            extract_audioLLDs('mfcc',self.aud_path)

        self.video_source.stop()
        self.audio_source.stop()
        
        self.vid_extract_thread = threading.Thread(target=extract_video_features)
        self.aud_extract_thread = threading.Thread(target=extract_audio_features)
        
        self.vid_extract_thread.start()
        self.aud_extract_thread.start()
    
        self.graph_frame.after(0, self.loading_animation, 0)

    def extract_emotion_values(self):
        audio_egemaps_fp = "audio_features/audio_features_egemaps/"+self.file_name+".csv"
        audio_mfcc_fp = "audio_features/audio_features_mfcc/"+self.file_name+".csv"
        visual_LLDS_fp = "visual_features_LLD/"+self.file_name+".csv"
        
        # Load in Features:
        audio_egemaps = summarise_audio(load_llds(audio_egemaps_fp))
        audio_mfcc = summarise_audio(load_llds(audio_mfcc_fp))
        visual_LLDS = load_llds(visual_LLDS_fp)
        
        min_seq_len = min((audio_egemaps.shape[0], audio_mfcc.shape[0], visual_LLDS.shape[0]))

        print("Obtaining Arousal Values ...")
        arousal_model = load_model("arousal.hdf5", custom_objects={'ccc_loss':ccc_loss, 'ContextVector':ContextVector}) 
        encoder_input = np.concatenate((audio_egemaps[:min_seq_len], audio_mfcc[:min_seq_len, :], visual_LLDS[:min_seq_len]),axis=-1)
    
        num_timesteps = min(encoder_input.shape[-2], 1768)
        if 1768-encoder_input.shape[0] >0 :
            encoder_input = np.concatenate((encoder_input, np.array([np.zeros(80) for _ in range(1768-encoder_input.shape[0])])), axis=0)
        elif 1768-encoder_input.shape[0] < 0:
            encoder_input = encoder_input[:1768]
        encoder_input = np.array([encoder_input])

        #arousal_vals = np.zeros(shape=(1,1768,1))
        #valence_vals = np.zeros(shape=(1,1768,1))
        #liking_vals = np.zeros(shape=(1,1768,1))

        #for i in range(num_timesteps):
        #    arousal_vals, valence_vals, liking_vals = arousal_model.predict_on_batch([encoder_input,arousal_vals, valence_vals, liking_vals])
        arousal_vals, valence_vals, liking_vals = arousal_model.predict_on_batch([encoder_input])
        arsl=[]
        vlnce = []
        lkng = []
        for i in range(num_timesteps):
            arsl.append(arousal_vals[0,i,0])
            vlnce.append(valence_vals[0,i,0])
            lkng.append(-1*liking_vals[0,i,0])

        self.arousal_values = pd.DataFrame(arsl)
        self.valence_values = pd.DataFrame(vlnce)
        self.liking_values = pd.DataFrame(lkng)
        

        """
        print("Obtaining Valence Values...")
        valence_model = load_model("valence.hdf5", custom_objects={'ccc_loss':ccc_loss, 'ContextVector':ContextVector})
        encoder_input = np.concatenate((audio_egemaps[:min_seq_len], audio_mfcc[:min_seq_len, :], visual_LLDS[:min_seq_len]),axis=-1)
    
        num_timesteps = min(encoder_input.shape[-2], 1768)
        if 1768-encoder_input.shape[0] >0 :
            encoder_input = np.concatenate((encoder_input, np.array([np.zeros(80) for _ in range(1768-encoder_input.shape[0])])), axis=0)
        elif 1768-encoder_input.shape[0] < 0:
            encoder_input = encoder_input[:1768]
        encoder_input = np.array([encoder_input])
        
        arousal_vals = np.zeros(shape=(1,1768,1))
        valence_vals = np.zeros(shape=(1,1768,1))
        liking_vals = np.zeros(shape=(1,1768,1))

        for i in range(num_timesteps):
            arousal_vals, valence_vals, liking_vals = valence_model.predict_on_batch([encoder_input,arousal_vals, valence_vals, liking_vals])
        
        vlnce=[]
        for i in range(num_timesteps):
            vlnce.append(valence_vals[0,i,0])
        print("Obtaining Liking Values...")
        arousal_vals = np.zeros(shape=(1,1768,1))
        valence_vals = np.zeros(shape=(1,1768,1))
        liking_vals = np.zeros(shape=(1,1768,1))

        for i in range(num_timesteps):
            arousal_vals, valence_vals, liking_vals = valence_model.predict_on_batch([encoder_input,arousal_vals, valence_vals, liking_vals])
        
        vlnce=[]
        for i in range(num_timesteps):
            vlnce.append(valence_vals[0,i,0])
        """
        """
        #self.valence_values = pd.DataFrame(vlnce)

        print("Obtaining Liking Values...")
        liking_model = load_model("liking.hdf5", custom_objects={'ccc_loss':ccc_loss, 'ContextVector':ContextVector})
        audio_encoder_input = np.concatenate((audio_egemaps[:min_seq_len], audio_mfcc[:min_seq_len, :]),axis=-1)
        visual_encoder_input = visual_LLDS[:min_seq_len]

        num_timesteps = min(audio_encoder_input.shape[-2], 1768)
        if 1768-audio_encoder_input.shape[0] >0 :
            audio_encoder_input = np.concatenate((audio_encoder_input, np.array([np.zeros(62) for _ in range(1768-audio_encoder_input.shape[0])])), axis=0)
            visual_encoder_input = np.concatenate((visual_encoder_input, np.array([np.zeros(18) for _ in range(1768-visual_encoder_input.shape[0])])), axis=0)
        elif 1768-audio_encoder_input.shape[0] < 0:
            audio_encoder_input = audio_encoder_input[:1768]
            visual_encoder_input = visual_encoder_input[:1768]
        
        audio_encoder_input = np.array([audio_encoder_input])
        visual_encoder_input = np.array([visual_encoder_input])

        arousal_vals = np.zeros(shape=(1,1768,1))
        valence_vals = np.zeros(shape=(1,1768,1))
        liking_vals = np.zeros(shape=(1,1768,1))

        for i in range(num_timesteps):
            arousal_vals, valence_vals, liking_vals = liking_model.predict_on_batch([audio_encoder_input,arousal_vals, valence_vals, liking_vals, visual_encoder_input])

    
        lkng = []
        for i in range(num_timesteps):
            lkng.append(liking_vals[0,i,0])

        self.liking_values = pd.DataFrame(lkng)
        """


if __name__ == "__main__":
    app = Application(Tk(), "EmotionAI")
    

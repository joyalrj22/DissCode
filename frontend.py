from __future__ import print_function, division
import numpy as np
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import wave

# BASED ON https://stackoverflow.com/questions/14140495/how-to-capture-a-video-and-audio-in-python-from-a-camera-or-webcamQ



class VideoRecorder():  

    def __init__(self, name="video/temp_video.avi", fourcc="MJPG", camindex=0, fps=25):
        self.video_out = None
        self.open = True
        self.device_index = camindex
        self.fps = fps                  # fps should be the minimum constant rate at which the camera can
        self.fourcc = fourcc            # capture images (with no decrease in speed over time; testing is required)
        self.video_filename = name    
        self.current_frame = (None, None)
        if not os.path.exists('video'):
            os.makedirs('video')


        self.video_cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        self.video_cap.open(self.device_index)
        if (self.video_cap.isOpened()==False):
            raise ValueError("Unable to open video source", camindex)

        self.sizex = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.sizey = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #frameSize = (sizex, sizey) # video formats and sizes also depend and vary according to the camera used

        self.frame_counts = 1
        self.start_time = time.time()
    
    def start_feed(self):
        while self.open:
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    self.current_frame = (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
                else:
                    self.current_frame = (ret, None)

    def record(self):
        "Video starts being recorded"

        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, (self.sizex, self.sizey))

        # counter = 1
        #timer_start = time.time()
        #timer_current = 0
        while self.open:
            ret, video_frame = self.current_frame
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            if ret:
                self.video_out.write(video_frame)
                #timer_current = time.time() - timer_start
                time.sleep(1/self.fps)
                #gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                """cv2.imshow('video_frame', gray)
                cv2.waitKey(1)"""
            else:
                break

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open and self.video_out:
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()
        self.open=False

    def delete(self):
        self.open = False
        if self.video_cap.isOpened():
            self.video_cap.release()

    def start(self, mode="feed"):
        if mode == "record":
            "Launches the video recording function using a thread"
            record_thread = threading.Thread(target=self.record)
            record_thread.start()
        elif mode == "feed":
            feed_thread = threading.Thread(target=self.start_feed)
            feed_thread.start()


class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="temp_audio.wav", rate=44100, fpb=1024, channels=2):
        if not os.path.exists('audio'):
            os.makedir('audio')
        self.stream = None
        self.filename = filename    
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.audio_file = self.create_file()
        self.audio_frames = []

    def create_file(self):
        f = wave.open(self.filename, 'wb')
        f.setnchannels(self.channels)
        f.setsampwidth(self.audio.get_sample_size(self.format))
        f.setframerate(self.rate)
        return f
    
    def stream_callback(self):
        def callback(inp, frame_count, time_info, status):
            self.audio_file.writeframes(inp)
            return inp, pyaudio.paContinue
        return callback

    def record(self):
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer, 
                                      stream_callback=self.stream_callback())
        self.stream.start_stream()
        
    def stop(self):
        if self.stream:
            self.stream.stop_stream()

    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

def start_AVrecording(filename="test"):
    global video_thread
    global audio_thread
    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()
    audio_thread.start()
    video_thread.start()
    return filename

def start_video_recording(filename="test"):
    global video_thread
    video_thread = VideoRecorder()
    video_thread.start()
    return filename

def start_audio_recording(filename="test"):
    global audio_thread
    audio_thread = AudioRecorder()
    audio_thread.start()
    return filename

def stop_AVrecording(filename="test"):
    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)

"""if __name__ == '__main__':
    start_AVrecording()
    time.sleep(5)
    stop_AVrecording()"""
    
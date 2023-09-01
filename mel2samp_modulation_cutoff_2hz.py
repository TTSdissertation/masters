# *****************************************************************************
#  Parent class in mel2samp.py at
#  https://github.com/NVIDIA/waveglow
# *****************************************************************************

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:15:01 2020

@author: js2251 and Laurel Ellis for master's project
"""

import random
import torch
import torch.utils.data
import sys
import argparse
import json
import os

from mel2samp import Mel2Samp, load_wav_to_torch, files_to_list
import numpy as np
import scipy.signal as ss

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

class Mel2SampModulationCutoff( Mel2Samp ):
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, dir_normal, dir_hi, target_factor = 1,n_mel_channels=80, f_cutoff = 2):
        super().__init__(training_files, segment_length, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax)        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax,n_mel_channels=n_mel_channels)
        self.dir_normal = dir_normal
        self.dir_hi     = dir_hi
        self.target_factor = target_factor
        self.f_cutoff = f_cutoff
            
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio_mel, sampling_rate = load_wav_to_torch(self.dir_normal + '/' + filename)
        audio, sampling_rate = load_wav_to_torch(self.dir_hi + '/' + filename)
        with torch.no_grad():
            audio *= self.target_factor
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
            audio_mel = audio_mel[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            audio_mel = torch.nn.functional.pad(audio_mel, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio_mel)
        
        mel = self.filter_modulation(mel,self.f_cutoff)
        
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)
    
    def filter_modulation(self,mel,cutoff_hz):
        b,a = ss.butter(3, cutoff_hz, 'lowpass', analog=False, fs=22050/256)
        for i in np.arange(mel.shape[0]):
            this_channel = mel[i,:]
            this_channel = ss.lfilter(b,a,this_channel)
            mel[i,:] = torch.from_numpy( this_channel )
        return mel
    
# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-a', "--a_cutoff", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    waveglow_config = json.loads(data)["waveglow_config"]
    mel2samp = Mel2SampModulationCutoff(**data_config,n_mel_channels=waveglow_config['n_mel_channels'], f_cutoff=args.a_cutoff)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        melspectrogram = mel2samp.filter_modulation(melspectrogram, 2)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
        
#runfile('mel2samp_two_files.py', args = '-f ' + file_wavs_filename + ' -o ' + mel_dir + ' -c config.json')
        
        
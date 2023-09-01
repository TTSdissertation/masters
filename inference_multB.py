# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:00:24 2020

@author: Antonia 
Used by Dr. Josef Schlittenlacher and Laurel Ellis for master's project
"""

import numpy as np
import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from my_denoiser import My_Denoiser
from convert_model import update_model

def inference_multiple(model_path, checkpoint_numbers, mel_files, output_dir, sigma=1.0,  sampling_rate=22050, is_fp16=False, denoiser_strength = 0, scale_out = 1):
    mel_files   = files_to_list(mel_files)
    
    for model_num in checkpoint_numbers:
        model_name = model_path + '/waveglow_' + str(model_num)
        print(model_name)
        waveglow = torch.load(model_name)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.cuda().eval()
        
        #output_dir_model = os.path.join( output_dir, str(model_num))
        #os.mkdir(output_dir_model)
        
        if is_fp16:
            from apex.amp import amp
            waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

        if denoiser_strength > 0:
            denoiser = My_Denoiser(waveglow,num_mel_bands=4).cuda()
            
        for i, file_path in enumerate(mel_files):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            mel = torch.load(file_path)
            #mel = torch.load(file_path, map_location='cpu')
            mel = torch.autograd.Variable(mel.cuda())
            #mel = torch.autograd.Variable(mel)
            if (mel.ndim < 3):
                mel = torch.unsqueeze(mel, 0)
            print(mel.shape)
            mel = mel.half() if is_fp16 else mel
            with torch.no_grad():
                audio = waveglow.infer(mel, sigma=sigma)
                if denoiser_strength > 0:
                    audio = denoiser(audio, denoiser_strength)
                audio = audio * scale_out
                audio = audio * MAX_WAV_VALUE
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            audio_path = os.path.join(
                output_dir, str(model_num) + '_' + file_name )
                #output_dir, str(model_num) + '_d' + str(int(denoiser_strength*10)) + '_' + file_name )
            write(audio_path, sampling_rate, audio)
            print(audio_path)
            
    


#tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
#torch.save(tacotron2, 'tacotron2_nvidia')
    
#waveglow_old = torch.load('models_nh/waveglow_585000')['model']
#waveglow = update_model(waveglow_old)
#torch.save(waveglow, 'models_nh/waveglow_585000')

inference_multiple('checkpoints16', [630000], 'mel_files16.txt', 'wav_files16', sigma=1.0,  sampling_rate=22050, is_fp16=False, denoiser_strength = 0, scale_out = 0.1)
#inference_multiple('D:/saved_models/waveglow/8chan', np.arange(940000,1000000,100000000000), 'mel_files_8_bands.txt', 'C:/python/waveglow/gen_sounds/8 mel bands', sigma=1.0,  sampling_rate=22050, is_fp16=False, denoiser_strength = 0, scale_out = 0.1)
#inference_multiple('D:/saved_models/waveglow/4chan_16flows', np.arange(450000,1000000,100000000000), 'mel_files_4_bands.txt', 'C:/python/waveglow/gen_sounds/4 mel bands 16 blocks', sigma=1.0,  sampling_rate=22050, is_fp16=False, denoiser_strength = 0, scale_out = 0.1)
#inference_multiple('C:/python/waveglow/checkpoints/4 bands', np.arange(1310000,10000000,100000000000), 'mel_files_4_bands.txt', 'C:/python/waveglow/gen_sounds/4 mel bands', sigma=1.0,  sampling_rate=22050, is_fp16=False, denoiser_strength = 0, scale_out = 0.1)




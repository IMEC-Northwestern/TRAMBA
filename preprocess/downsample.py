import operator
import os
import librosa
import numpy as np
import soundfile as sf
import torch
import random
import numpy as np
import sox
from scipy.signal import decimate
from torchaudio.functional import resample

def pad(sig, length):

    if len(sig) < length:
        pad = length - len(sig)
        sig = np.hstack((sig, np.zeros(pad) + 0.1))
    else:
        sig = sig[:length]

    return sig

def lowpass(sig, file_path):
    
    low_sr = 16000 // 4
    sig = sig[::4]
    sig_tensor = torch.from_numpy(sig)
    upsampled_tensor = resample(sig_tensor, low_sr, 16000)
    sig = upsampled_tensor.numpy()

    return sig


def process_audio_file(file_path, sr, window, stride):

    sig, _ = librosa.load(file_path, sr=sr)
    if len(sig) < window:
        sig = pad(sig, window)
    batches = int((len(sig) - stride) / stride)
    sig = sig[0: int(batches * stride + stride)]
    target = sig.copy()   
    low_sig = lowpass(sig, file_path)   
    if len(target) != len(low_sig):
        low_sig = pad(low_sig, len(target))

    return target, low_sig

def process_folder(folder_path, sr, window, stride, save_path_base):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                target, low_sig = process_audio_file(file_path, sr, window, stride)
                
                # Define save paths for target and low_sig
                target_save_path = os.path.join(save_path_base, 'target', os.path.relpath(root, folder_path), file)
                low_sig_save_path = os.path.join(save_path_base, 'low_sig', os.path.relpath(root, folder_path), file)
                
                # Make sure the catalog exists
                os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(low_sig_save_path), exist_ok=True)
                
                # Save target and low_sig, respectively.
                sf.write(target_save_path, target, sr)
                sf.write(low_sig_save_path, low_sig, sr)
def main():

    sr = 16000  
    window = 8192  
    stride = 4096  

    folder_path = 'Here is the original audio path'
    save_path = 'Here is the saving path'

    process_folder(folder_path, sr, window, stride, save_path)

if __name__ == "__main__":
    main()

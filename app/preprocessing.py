import numpy as np
import librosa

sample_rate=22050
duration=3
samples=sample_rate*duration
n_mfcc=40
n_mels=128
max_frames=130

def extract_mfcc(filepath:str)->np.ndarray:
    audio,sr=librosa.load(filepath,sr=sample_rate,duration=3,res_type="soxr_hq")
    if len(audio)<samples:
        audio=np.pad(audio,(0,samples-len(audio)),mode="constant")
    else:
        audio=audio[:samples]
    

    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=n_mfcc)
    mfcc=(mfcc-np.mean(mfcc,axis=0))/(np.std(mfcc,axis=0)+1e-6)

    if mfcc.shape[1]<max_frames:
        mfcc=np.pad(mfcc,((0,0),(0,max_frames-mfcc.shape[1])))
    else:
        mfcc=mfcc[:,:max_frames]
    
    mfcc=mfcc.T
    return mfcc

def validate_audio(filepath:str)->None:
    import soundfile as sf
    import os
    if not os.path.exists(filepath):
        raise ValueError("Audio File not found.")
    info=sf.info(filepath)

    if info.duration<1.0:
        raise ValueError(f"Audio too short{info.duration:.1f}s. Minimum 1 seconds required.")
    if info.duration>60:
        raise ValueError(f"Audio too long {info.duration:.1f}s. Maximum 60 seconds accepted.")
    
    
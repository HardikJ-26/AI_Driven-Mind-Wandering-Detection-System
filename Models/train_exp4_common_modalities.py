"""Train a Transformer on Exp4 using only EEG, gaze/pupil/blink, and heart rate."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiment4' / 'RESULTS' / 'exp4_common_modalities_transformer.pkl'
STEPS, WINDOWS, EPOCHS = 40, 30, 25

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.conv=nn.Conv1d(channels,64,3,padding=1); self.t=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,batch_first=True),2)
    def forward(self,x): return self.t(F.relu(self.conv(x.transpose(1,2))).transpose(1,2))
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.e,self.o,self.p=Encoder(64),Encoder(4),Encoder(1); self.ao=nn.MultiheadAttention(64,4,batch_first=True); self.ap=nn.MultiheadAttention(64,4,batch_first=True); self.h=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,e,o,p):
        e,o,p=self.e(e),self.o(o),self.p(p); a=self.ao(e,o,o)[0]; b=self.ap(e,p,p)[0]; return self.h(torch.cat((a,b),-1).mean(1))
def rs(x,n=STEPS*WINDOWS):
    x=np.asarray(x,dtype='float32'); x=x[:,None] if x.ndim==1 else x; x=pd.DataFrame(x).interpolate(limit_direction='both').ffill().bfill().to_numpy(); a=np.linspace(0,1,len(x)); b=np.linspace(0,1,n); return np.column_stack([np.interp(b,a,x[:,i]) for i in range(x.shape[1])]).astype('float32')
def f(root,sub,ses,stim,mod,desc):
    q=list((root/sub/ses/mod).glob(f'{sub}_{ses}_task-stim{stim:02d}_desc-{desc}*')); return q[0] if q else None
def load(root, labels, max_stim):
    E=[];O=[];P=[];M=[]
    for sub,y in labels.items():
      for ses in ('ses-01','ses-02'):
       for stim in range(1,max_stim+1):
        eeg=f(root,sub,ses,stim,'eeg','eeg.bdf'); g=f(root,sub,ses,stim,'eyetrack','gaze_visualangle_eyetrack.tsv'); u=f(root,sub,ses,stim,'eyetrack','pupil_eyetrack.tsv'); b=f(root,sub,ses,stim,'eyetrack','blinkrate.tsv'); h=f(root,sub,ses,stim,'beh','heartrate.tsv')
        if not all((eeg,g,u,b,h)): continue
        try:
         e=rs(mne.io.read_raw_bdf(eeg,preload=True,verbose=False).get_data().T); gaze=rs(np.loadtxt(g,delimiter='\t',ndmin=2)[:,:2]); o=np.c_[gaze,rs(np.loadtxt(u,delimiter='\t',ndmin=2)[:,0]),rs(np.loadtxt(b,delimiter='\t',ndmin=2)[:,0])]; p=rs(np.loadtxt(h,delimiter='\t',ndmin=2)[:,0])
         for w in range(WINDOWS): E.append(e[w*STEPS:(w+1)*STEPS]);O.append(o[w*STEPS:(w+1)*STEPS]);P.append(p[w*STEPS:(w+1)*STEPS]);M.append((sub,int(y)))
        except Exception: pass
    return np.stack(E),np.stack(O),np.stack(P),M
labels=pd.read_csv(ROOT/'experiment4'/'RESULTS'/'master_encoded_labels.csv').drop_duplicates('participant_id').set_index('participant_id').attention_label
E,O,P,M=load(ROOT/'experiment4'/'derivatives',labels,6); y=np.array([x[1] for x in M],dtype='float32'); stats=[]
for x in (E,O,P):
 m=x.mean((0,1),keepdims=True); s=x.std((0,1),keepdims=True);s[s<1e-7]=1; x-=m;x/=s;stats.append((m,s))
model=Model(); opt=torch.optim.Adam(model.parameters(),lr=1e-3); pos=(len(y)-y.sum())/y.sum()
for _ in range(EPOCHS):
 for i in torch.randperm(len(y)).split(16):
  q=i.numpy();z=model(torch.from_numpy(E[q]),torch.from_numpy(O[q]),torch.from_numpy(P[q]));t=torch.from_numpy(y[q,None]);loss=-(pos*t*torch.log(z.clamp(1e-7,1-1e-7))+(1-t)*torch.log(1-z.clamp(1e-7,1-1e-7))).mean();opt.zero_grad();loss.backward();opt.step()
with open(OUT,'wb') as h: pickle.dump({'state_dict':model.state_dict(),'normalization':stats,'modalities':['EEG-64','gaze_x','gaze_y','pupil','blink_rate','heart_rate'],'window_seconds':10,'labels':'ASRS screener > 4'},h)
print(f'Saved {OUT}')

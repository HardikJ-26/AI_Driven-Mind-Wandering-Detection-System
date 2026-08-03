"""Cross-experiment test: loads Exp4 checkpoint and evaluates Exp2 ASRS labels."""
import pickle, runpy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
ns=runpy.run_path(Path(__file__).with_name('train_exp4_common_modalities.py'),run_name='transfer_helpers')
ROOT=Path(__file__).resolve().parents[2]; out=ROOT/'experiment2'/'cross_model';out.mkdir(exist_ok=True)
with open(ROOT/'experiment4'/'RESULTS'/'exp4_common_modalities_transformer.pkl','rb') as h: ckpt=pickle.load(h)
labels=pd.read_csv(ROOT/'experiment2'/'phenotype'/'asrs_questionnaire.tsv').set_index('participant_id').dichotomous_screener_score.gt(4).astype(int)
E,O,P,M=ns['load'](ROOT/'experiment2'/'derivatives',labels,5)
for x,(m,s) in zip((E,O,P),ckpt['normalization']): x[:]=(x-m)/s
model=ns['Model']();model.load_state_dict(ckpt['state_dict']);model.eval()
with torch.no_grad(): prob=model(torch.from_numpy(E),torch.from_numpy(O),torch.from_numpy(P)).squeeze().numpy()
d=pd.DataFrame({'participant_id':[x[0] for x in M],'label':[x[1] for x in M],'probability':prob}).groupby('participant_id').mean();d['prediction']=(d.probability>=.5).astype(int);d.to_csv(out/'exp4_to_exp2_predictions.csv')
y,p=d.label.values,d.prediction.values; metrics={'accuracy':accuracy_score(y,p),'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'roc_auc':roc_auc_score(y,d.probability) if len(set(y))==2 else np.nan}
pd.DataFrame(metrics.items(),columns=['metric','value']).to_csv(out/'exp4_to_exp2_metrics.csv',index=False)
(out/'exp4_to_exp2_report.md').write_text('# Exp4 → Exp2 Transfer Report\n\nCommon modalities: 64-channel EEG, gaze X/Y, pupil, blink rate, and heart rate. EOG and respiration were excluded.\n\n'+'\n'.join(f'- {k}: {v:.4f}' for k,v in metrics.items())+'\n\nInterpretation: these are participant-level ASRS-screener results. Cross-experiment performance may be reduced by task, device, and distribution differences; do not claim clinical validity from this evaluation.\n')
print(metrics)

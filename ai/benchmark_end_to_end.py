from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset' / 'v5_1000'
MODEL_DIR = ROOT / 'ai' / 'models'
RESULT_DIR = ROOT / 'ai' / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['soc','soh','current_A','initial_temperature_C']
TARGET = 'T_max_C'

class V5ThermalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.network(x)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
    def forward(self,x): return self.network(x)

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    meta = pd.read_csv(DATASET/'metadata.csv')
    thermal = pd.read_csv(DATASET/'thermal_metrics.csv')
    df = meta.merge(thermal,on='case_id',how='inner')
    df = df[df['valid']==1].copy()
    idx=np.arange(len(df))
    train_idx,test_idx=train_test_split(idx,test_size=0.20,random_state=42)
    train=df.iloc[train_idx].copy(); test=df.iloc[test_idx].copy().reset_index(drop=True)
    xs=StandardScaler().fit(train[FEATURES].values.astype(np.float32))
    ys=StandardScaler().fit(train[[TARGET]].values.astype(np.float32))
    X=torch.from_numpy(xs.transform(test[FEATURES].values.astype(np.float32))).to('cuda')
    base=V5ThermalMLP(); base.load_state_dict(torch.load(MODEL_DIR/'v5_tmax_mlp.pt',map_location='cpu')); base=base.to('cuda').eval()
    residual=ResidualMLP(); residual.load_state_dict(torch.load(MODEL_DIR/'v5_residual_mlp.pt',map_location='cpu')); residual=residual.to('cuda').eval()
    with torch.no_grad():
        for _ in range(20): base(X); residual(X)
    torch.cuda.synchronize()
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.no_grad(): bp=base(X)
    torch.cuda.synchronize(); base_ms=(time.perf_counter()-t)*1000
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.no_grad(): bp2=base(X); rp=residual(X)
    torch.cuda.synchronize(); hybrid_ms=(time.perf_counter()-t)*1000
    bp=ys.inverse_transform(bp.detach().cpu().numpy()).ravel()
    hp=bp2.detach().cpu().numpy(); hp=ys.inverse_transform(hp).ravel()+rp.detach().cpu().numpy().ravel()
    actual=test[TARGET].values.astype(np.float32)
    physics_ms=float(test['runtime_ms'].sum())
    out=pd.DataFrame([
      ['V5 Physics CUDA',len(test),np.nan,np.nan,np.nan,physics_ms,physics_ms/len(test),1.0,1.0],
      ['Base AI GPU',len(test),np.sqrt(mean_squared_error(actual,bp)),mean_absolute_error(actual,bp),r2_score(actual,bp),base_ms,base_ms/len(test),base_ms/physics_ms,physics_ms/base_ms],
      ['Hybrid AI GPU',len(test),np.sqrt(mean_squared_error(actual,hp)),mean_absolute_error(actual,hp),r2_score(actual,hp),hybrid_ms,hybrid_ms/len(test),hybrid_ms/physics_ms,physics_ms/hybrid_ms]
    ],columns=['method','test_cases','RMSE_C','MAE_C','R2','total_runtime_ms','runtime_per_case_ms','relative_runtime_vs_physics','physics_runtime_speedup'])
    out.to_csv(RESULT_DIR/'final_end_to_end_benchmark.csv',index=False)
    detail=test[['case_id']+FEATURES+[TARGET,'runtime_ms']].copy(); detail['base_ai_Tmax_C']=bp; detail['hybrid_ai_Tmax_C']=hp; detail['base_ai_error_C']=bp-actual; detail['hybrid_ai_error_C']=hp-actual
    detail.to_csv(RESULT_DIR/'final_end_to_end_test_cases.csv',index=False)
    print(out.to_string(index=False)); print('\nSaved:',RESULT_DIR/'final_end_to_end_benchmark.csv'); print(RESULT_DIR/'final_end_to_end_test_cases.csv')

if __name__=='__main__': main()

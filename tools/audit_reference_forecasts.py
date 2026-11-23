from __future__ import annotations
import ast,hashlib,json,math,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code/model_nowcasting_hibrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end.py'
DATA=ROOT/'data/CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip'
OUT=ROOT/'generated_audit'; OUT.mkdir(exist_ok=True)
tree=ast.parse(CODE.read_text(encoding="utf-8-sig"))
wanted={'time_slot_of_day','valid_t_end_indices_per_segment','climatology_lookup','build_baseline_climatology','robust_recent_temperature_slope','diurnal_temperature_coherence','damped_temperature_trend_prior','_ensure_naive_utc','solar_geometry_features','get_time_feature_columns','make_time_features'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WindowDataset')
nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_temp_baseline']
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
ns={'np':np,'pd':pd,'math':math}
exec(compile(ast.fix_missing_locations(module),str(CODE),'exec'),ns)
cols={'TIMESTAMP','Temp_1_Avg','Lluvia_Tot','segment_id','valid_t_end','Rad_Global_Avg','Temp_1_Avg__miss','Lluvia_Tot__miss'}
with zipfile.ZipFile(DATA) as z:
    csv=z.namelist()[0]
    with z.open(csv) as f: df=pd.read_csv(f,usecols=lambda x:x in cols)
df.TIMESTAMP=pd.to_datetime(df.TIMESTAMP)
df=df.sort_values('TIMESTAMP').drop_duplicates('TIMESTAMP',keep='last').set_index('TIMESTAMP')
cfg=SimpleNamespace(freq='10min',temp_col='Temp_1_Avg',rain_col='Lluvia_Tot',segment_col='segment_id',rain_event_threshold=0.1,accumulation_horizons_hours=(1.,2.,3.),horizon_hours=3.)
train=df.loc[df.index<'2023-01-01'];test=df.loc[(df.index>='2024-01-01')&(df.index<'2025-01-01')]
valid=ns['valid_t_end_indices_per_segment'](test,cfg,288,18)
print('Loaded',len(df),'rows;',len(valid),'test origins',flush=True)
ctx=ns['build_baseline_climatology'](train,cfg,6)
ctx={k:np.asarray(v,dtype=np.float64) if isinstance(v,list) else v for k,v in ctx.items()}
months=test.index.month.to_numpy(dtype=np.int64);slots=ns['time_slot_of_day'](test.index,'10min')
featnames=ns['get_time_feature_columns'](True)
tf=ns['make_time_features'](test.index,latitude_deg=41.385020,longitude_deg=2.117763,use_solar_geometry=True)
cosz=tf[:,featnames.index('solar_cos_zenith')]
toa=1361.0*np.clip(tf[:,featnames.index('solar_toa_sw')],0,None)
T=test.Temp_1_Avg.to_numpy(np.float32); R=test.Lluvia_Tot.to_numpy(np.float32)
seg=test.segment_id.to_numpy(np.int64); futures=valid[:,None]+np.arange(1,19)
y=T[futures]; yr=R[futures]
mask=np.ones_like(y,dtype=bool) if 'Temp_1_Avg__miss' not in test else (1-test['Temp_1_Avg__miss'].to_numpy(np.float32)[futures])>0.5
maskr=np.ones_like(yr,dtype=bool) if 'Lluvia_Tot__miss' not in test else (1-test['Lluvia_Tot__miss'].to_numpy(np.float32)[futures])>0.5

def metrics(pred,truth=y,m=mask):
    d=np.asarray(pred,dtype=np.float64)-truth.astype(np.float64)
    count=m.sum(axis=0)
    mae=np.where(m,abs(d),0).sum(axis=0)/count
    rmse=np.sqrt(np.where(m,d*d,0).sum(axis=0)/count)
    bias=np.where(m,d,0).sum(axis=0)/count
    return {'mae_mean':float(mae.mean()),'rmse_mean':float(rmse.mean()),'bias_mean':float(bias.mean()),'mae_by_lead':mae.tolist(),'rmse_by_lead':rmse.tolist(),'valid_counts_by_lead':count.tolist()}

preds={'persistence':np.broadcast_to(T[valid,None],y.shape)}
preds['standalone_month_slot_climatology']=ns['climatology_lookup'](months[futures],slots[futures],ctx['temp_month_slot'],ctx['temp_slot'],ctx['temp_global']).astype(np.float32)
prev=futures-144
prev_values=T[prev].copy(); good=np.all(seg[prev]==seg[valid,None],axis=1)
prev_values[~good]=np.nan
preds['previous_day_same_time']=np.where(np.isfinite(prev_values),prev_values,preds['standalone_month_slot_climatology'])
obj=SimpleNamespace(bc=ctx,O=18,steps_per_hour=6.,use_temp_trend_baseline=True,temp_trend_short_hours=1.,temp_trend_mid_hours=3.,temp_trend_long_hours=6.,temp_trend_max_abs_slope_C_per_hour=2.,temp_trend_baseline_weight=.15,temp_trend_tau_hours=1.75,temp_trend_max_delta_C=1.25,solar_temp_baseline_weight=.25)
hybrid=np.empty_like(y)
for k,t in enumerate(valid):
    hybrid[k],_=ns['_temp_baseline'](obj,float(T[t]),months[futures[k]],slots[futures[k]],prev_values[k],cosz[futures[k]],toa[futures[k]],T[t-287:t+1],float(toa[t]))
    if k and k%15000==0:print('Hybrid reference',k,'/',len(valid),flush=True)
preds['hybrid_temperature_reference']=hybrid
summary={name:metrics(p) for name,p in preds.items()}
# The training-only Markov comparator needs no learned neural parameters.
occ=np.empty(yr.shape,np.float32);mu=np.empty(yr.shape,np.float32)
pp=(R[valid]>=.1).astype(np.float64)
for j in range(18):
    m=months[futures[:,j]];s=slots[futures[:,j]]
    p01=ns['climatology_lookup'](m,s,ctx['markov_p01_month_slot'],ctx['markov_p01_slot'],ctx['markov_p01_global'],0,1)
    p11=ns['climatology_lookup'](m,s,ctx['markov_p11_month_slot'],ctx['markov_p11_slot'],ctx['markov_p11_global'],0,1)
    pp=pp*p11+(1-pp)*p01;occ[:,j]=pp
    wet=ns['climatology_lookup'](m,s,ctx['rain_amt_wet_month_slot'],ctx['rain_amt_wet_slot'],ctx['rain_amt_wet_global'],0,None)
    mu[:,j]=pp*wet
summary['rain_markov']=metrics(mu,yr,maskr)
briers=np.where(maskr,(occ-(yr>=.1))**2,0).sum(axis=0)/maskr.sum(axis=0)
summary['rain_markov']['brier_mean']=float(briers.mean())
summary['rain_markov']['brier_by_lead']=briers.tolist()
report={'scope':'Independent deterministic-reference audit; not a neural-model reproduction. Source-script defaults and explicit supplied launch parameters; timestamps unchanged.','source_sha256':hashlib.sha256(CODE.read_bytes()).hexdigest(),'dataset_zip_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'test_origins':len(valid),'test_first_origin':str(test.index[valid[0]]),'test_last_origin':str(test.index[valid[-1]]),'temperature_references':{k:v for k,v in summary.items() if k!='rain_markov'},'rain_markov':summary['rain_markov']}
(OUT/'reference_recalculation.json').write_text(json.dumps(report,indent=2))
for name,m in summary.items():print(name,{k:v for k,v in m.items() if k.endswith('_mean')},flush=True)
# Save temperature references only; no model predictions are manufactured.
np.savez_compressed(OUT/'temperature_reference_predictions.npz',origin_timestamp=test.index[valid].to_numpy(dtype='datetime64[ns]'),observed_temperature=y,valid_temperature_mask=mask,**preds)
# Inspect solar alignment: shift hypotheses are sensitivity diagnostics, not metadata proof.
solar_audit={}
for season,months_selected in [('DJF',[12,1,2]),('JJA',[6,7,8])]:
    d=df.loc[df.index.month.isin(months_selected)]
    rad=d.Rad_Global_Avg.to_numpy(float)
    phase=d.index.hour.to_numpy()*60+d.index.minute.to_numpy()
    high=pd.DataFrame({'phase':phase,'rad':rad}).groupby('phase').rad.quantile(.95)
    solar_audit[season]={'p95_radiation_peak_label':f'{int(high.idxmax())//60:02d}:{int(high.idxmax())%60:02d}','peak_w_m2':float(high.max())}
night=[]
for name,offset in [('UTC',0),('fixed_CET_UTC_plus_1',60),('Europe_Madrid_civil',None)]:
    idx=df.index
    if offset is None:
        # Treat ambiguous autumn times both ways as NaT: exclude those cases rather than invent folds.
        utc=idx.tz_localize('Europe/Madrid',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC').tz_localize(None)
    else:utc=idx-pd.Timedelta(minutes=offset)
    finite=~utc.isna(); sun=ns['solar_geometry_features'](utc[finite],41.385020,2.117763)['cos_zenith']
    rad=df.Rad_Global_Avg.to_numpy(float)[finite]
    strong=rad>100
    deep_night=sun<np.sin(np.deg2rad(-6.))
    night.append({'hypothesis':name,'records':int(finite.sum()),'radiation_above_100_records':int(strong.sum()),'above_100_at_sun_below_minus6_degrees':int((strong&deep_night).sum()),'fraction_of_strong_radiation_in_deep_night':float((strong&deep_night).sum()/strong.sum())})
(OUT/'timestamp_diagnostics.json').write_text(json.dumps({'warning':'Solar consistency diagnostics cannot certify historical logger clock metadata or averaging-interval conventions.','seasonal_p95_peaks':solar_audit,'hypotheses':night},indent=2))
print('Solar diagnostics',json.dumps(solar_audit),night,flush=True)

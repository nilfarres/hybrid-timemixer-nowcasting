import ast,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
src=root/'code/model_nowcasting_hibrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end.py'
nodes=[n for n in ast.parse(src.read_text(encoding="utf-8-sig")).body if isinstance(n,ast.FunctionDef) and n.name in {'_ensure_naive_utc','solar_geometry_features'}]
ns={'pd':pd,'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(src),'exec'),ns)
with zipfile.ZipFile(root/'data/CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip') as z:
 with z.open(z.namelist()[0]) as f:d=pd.read_csv(f,usecols=['TIMESTAMP','Rad_Global_Avg'],parse_dates=['TIMESTAMP'])
d=d.set_index('TIMESTAMP'); rows=[]
for year in range(2011,2025):
 for month in [1,7]:
  a=d.loc[(d.index.year==year)&(d.index.month==month)];a=a.loc[np.isfinite(a.Rad_Global_Avg)]
  if len(a)<1000:continue
  slots=(a.index.hour*6+a.index.minute//10).to_numpy()
  q=a.groupby(slots).Rad_Global_Avg.quantile(.95)
  if len(q)<140:continue
  observed=q.reindex(range(144)).to_numpy(); fits=[]
  for delta in range(-180,181,10):
   toa=ns['solar_geometry_features'](a.index+pd.Timedelta(minutes=delta),41.385020,2.117763)['toa_sw']
   curve=pd.Series(toa).groupby(slots).mean().reindex(range(144)).to_numpy()
   ok=np.isfinite(observed)&np.isfinite(curve)
   # Least-squares scale only; solar forcing and measured radiation need not share an amplitude.
   scale=float(np.dot(observed[ok],curve[ok])/np.dot(curve[ok],curve[ok]))
   rmse=float(np.sqrt(np.mean((observed[ok]-scale*curve[ok])**2)))
   fits.append({'utc_offset_applied_minutes':delta,'rmse_w_m2':rmse,'scale':scale})
  best=min(fits,key=lambda x:x['rmse_w_m2'])
  rows.append({'year':year,'month':month,'best_offset_applied_minutes':best['utc_offset_applied_minutes'],'best_rmse_w_m2':best['rmse_w_m2'],'utc_rmse_w_m2':next(x['rmse_w_m2'] for x in fits if x['utc_offset_applied_minutes']==0),'cet_rmse_w_m2':next(x['rmse_w_m2'] for x in fits if x['utc_offset_applied_minutes']==-60),'cest_rmse_w_m2':next(x['rmse_w_m2'] for x in fits if x['utc_offset_applied_minutes']==-120)})
result={'method':'Monthly 95th-percentile radiation curve, January and July in each year. Mean TOA curve from supplied solar routine with label shifts -180..180 min (10-min increments); least-squares amplitude; all 144 slots. Diagnostic evidence, not logger metadata certification.','results':rows,'median_best_offset_minutes':float(np.median([x['best_offset_applied_minutes'] for x in rows]))}
p=root/'generated_audit/solar_alignment.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

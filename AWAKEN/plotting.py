"""Publication figures from the results produced by main.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import FONT_FAMILY, use_times_new_roman
use_times_new_roman()
import json
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
MODELS=['top_down','turbopark','gaussian','array_stability','no_wake']
LABEL={'top_down':'Onshore top down','td_offshore_transfer':'Original top down',
       'turbopark':'TurboPark','gaussian':'Gaussian','array_stability':'ASM + TurboPark','no_wake':'No-wake reference'}
COLORS=dict(zip(MODELS,['#1768ac','#008c75','#d97809','#a84992','#666666']))
COLORS['td_offshore_transfer']='#8b739f'

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)

def summarize(df):
    primary=df[df.model.isin(MODELS)]
    summary=primary.groupby('model').agg(cases=('case','size'),
        field_mae_pp=('field_mae_pp','mean'),field_rmse_pp=('field_rmse_pp','mean'),
        profile_mae_pp=('profile_mae_pp','mean'),profile_cases=('profile_mae_pp','count'),
        integral_mae_km=('integral_mae_km','mean'),integral_cases=('integral_mae_km','count'))
    summary.to_csv(TABLE/'awaken_main_metrics.csv')
    df.groupby('model').agg(cases=('case','size'),field_mae_pp=('field_mae_pp','mean'),
        field_rmse_pp=('field_rmse_pp','mean'),profile_mae_pp=('profile_mae_pp','mean'),
        profile_cases=('profile_mae_pp','count')).to_csv(TABLE/'awaken_all_configurations.csv')
    primary.to_csv(TABLE/'awaken_case_metrics.csv',index=False)
    td=df[df.model=='top_down'].set_index('case')
    wide=primary.pivot(index='case',columns='model',values='field_mae_pp')
    dates=td.start.astype(str).str[:10].reindex(wide.index)
    rng=np.random.default_rng(20260909)
    unique=dates.unique()
    paired=[]
    for model in MODELS[1:]:
        delta=wide.top_down-wide[model]
        groups=[delta[dates==day].to_numpy() for day in unique]
        bootstrap=np.array([np.concatenate([groups[i] for i in sample]).mean()
            for sample in rng.integers(0,len(groups),(10000,len(groups)))])
        lo,hi=np.quantile(bootstrap,[.025,.975])
        paired.append(dict(comparator=model,mean_difference_pp=delta.mean(),
            median_difference_pp=delta.median(),ci_low_pp=lo,ci_high_pp=hi,
            top_down_lower_cases=int((delta<0).sum()),dates=len(unique)))
    pd.DataFrame(paired).to_csv(TABLE/'awaken_paired_differences.csv',index=False)
    grouping=[]
    for column in ['stability','complexity','farm']:
        for (level,model),g in primary.groupby([column,'model']):
            grouping.append(dict(grouping=column,group=level,model=model,cases=len(g),
                                 field_mae_pp=g.field_mae_pp.mean()))
    pd.DataFrame(grouping).to_csv(TABLE/'awaken_groups.csv',index=False)
    primary[~primary.complexity.str.contains('direction',case=False)].groupby('model').agg(
        cases=('case','size'),field_mae_pp=('field_mae_pp','mean')).to_csv(TABLE/'awaken_direction_qc_sensitivity.csv')
    sensitivity=[]
    for name,g in df[~df.model.isin(MODELS)].groupby('model'):
        aligned=g.set_index('case').field_mae_pp-td.field_mae_pp
        sensitivity.append(dict(scenario=name,cases=len(g),field_mae_pp=g.field_mae_pp.mean(),
            mean_change_pp=aligned.mean(),mean_absolute_case_change_pp=aligned.abs().mean(),
            maximum_absolute_case_change_pp=aligned.abs().max()))
    pd.DataFrame(sensitivity).to_csv(TABLE/'awaken_input_sensitivity.csv',index=False)
    return summary,td,wide,pd.DataFrame(paired)

def context(td,wide,paired):
    fig,axs=plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
    inv=pd.read_csv(RESULT/'turbine_mapping.csv')
    for farm,g in inv.groupby('p_name'):
        axs[0,0].scatter(g.xlong,g.ylat,s=8,label=farm.replace('unknown ','')+f' ({len(g)})')
    axs[0,0].set(xlabel='Longitude (°)',ylabel='Latitude (°)',title='(a) Supplied turbine inventory')
    axs[0,0].legend(fontsize=7,loc='lower left')
    axs[0,0].xaxis.set_major_locator(MaxNLocator(5))
    axs[0,0].set_aspect(1/np.cos(np.deg2rad(36.3)))
    for label,g in td.groupby('stability'):
        axs[0,1].scatter(80/g.L_m,g.wind_speed_ms,s=27,label=f'{label} ({len(g)})')
    axs[0,1].axvline(0,color='.6',lw=.8)
    axs[0,1].set(xlabel='Stability coordinate, 80 m / L',ylabel='Radar reference speed (m s$^{-1}$)',
                 title='(b) Fixed 35-period subset')
    axs[0,1].legend(fontsize=8)
    all_scores=pd.read_csv(RESULT/'metrics.csv').pivot(index='case',columns='model',values='field_mae_pp')
    shown=['td_offshore_transfer',*MODELS]
    for j,m in enumerate(shown):
        axs[1,0].scatter(np.full(len(wide),j)+np.linspace(-.15,.15,len(wide)),all_scores[m],
                         s=12,alpha=.60,color=COLORS.get(m,'#8b739f'))
        axs[1,0].plot(j,all_scores[m].mean(),'D',color='black',ms=5)
    axs[1,0].set(xticks=range(len(shown)),xticklabels=['Original\ntop down','Onshore\ntop down','TurboPark','Gaussian','ASM +\nTurboPark','No wake'],
                 ylabel='Case field MAE (pp)',title='(c) All cases; diamonds: mean')
    axs[1,0].tick_params(axis='x',labelsize=8)
    for j,r in paired.iterrows():
        axs[1,1].plot([r.ci_low_pp,r.ci_high_pp],[j,j],color=COLORS[r.comparator],lw=2)
        axs[1,1].plot(r.mean_difference_pp,j,'o',color=COLORS[r.comparator])
    axs[1,1].axvline(0,color='.4',lw=.8)
    axs[1,1].set(yticks=range(len(MODELS)-1),yticklabels=[LABEL[m] for m in MODELS[1:]],
        xlabel='Top-down minus comparator MAE (pp)',title='(d) Paired mean; 95% date-block interval')
    for ax in axs.ravel():ax.spines[['top','right']].set_visible(False)
    td.reset_index().to_csv(DATA/'awaken_context.csv',index=False)
    wide.to_csv(DATA/'awaken_case_errors.csv')
    save(fig,'awaken_context_and_scores')

def representative(td, cluster_fields=None):
    """Figure 8 uses a full-cluster field separate from downstream scoring."""
    from cluster_figure import draw
    fields=Path(cluster_fields) if cluster_fields else Path(__file__).resolve().parent/'data/cluster_figure/reference.npz'
    with np.load(fields) as z:meta=json.loads(str(z['metadata']))
    if meta['case'] not in td.index:raise ValueError('Cluster figure case is outside benchmark')
    return draw(fields,FIG,DATA)


def make_figures(metrics, fields, output, cluster_fields=None):
    global FIG, TABLE, DATA, RESULT, FIELD_DIR
    FIG=output/'figures';TABLE=output/'tables';DATA=output/'figure_data'
    RESULT=output;FIELD_DIR=fields
    for p in [FIG,TABLE,DATA]:p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':11,'axes.titlesize':11,'axes.linewidth':.7})
    summary,td,wide,paired=summarize(metrics)
    context(td,wide,paired);representative(td,cluster_fields)

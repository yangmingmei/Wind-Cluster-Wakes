"""Publication figures from the results produced by main.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import FONT_FAMILY, use_times_new_roman
use_times_new_roman()
MODELS = ['top_down', 'turbopark', 'gaussian', 'array_stability']
LABEL = {'wrf': 'WRF', 'observed': 'WIPAFF', 'top_down': 'Top-down',
         'turbopark': 'TurboPark', 'gaussian': 'Gaussian', 'array_stability': 'Array-stability + TurboPark'}
COLOR = {'observed': '#1b252e', 'top_down': '#1768ac', 'gaussian': '#d97809',
         'turbopark': '#008c75', 'array_stability': '#8b56a5'}
STYLE = {'observed': '-', 'top_down': '-', 'gaussian': '--', 'turbopark': '-.', 'array_stability': ':'}

def save(fig, name):
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=320, bbox_inches='tight')
    plt.close(fig)
def make_figures(profiles, input_dir, output):
    global OUT, DATA
    OUT=output/"figures"; DATA=output/"figure_data"
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    cases = sorted(profiles.case_id.unique())
    plot_rows, peak_rows = [], []
    fig, axes = plt.subplots(2,5,figsize=(10.4,5.5),sharex=True,sharey=True)
    for row, case in enumerate(cases):
        for col, leg in enumerate(range(1,6)):
            ax=axes[row,col]
            observed=profiles.loc[(profiles.case_id==case)&(profiles.leg_id==leg)&(profiles.model=='top_down')].sort_values('crosswind_bin_km')
            distance=float(observed.actual_downstream_distance_km.iloc[0])
            for m in ['observed',*MODELS]:
                g=observed if m=='observed' else profiles.loc[(profiles.case_id==case)&(profiles.leg_id==leg)&(profiles.model==m)].sort_values('crosswind_bin_km')
                raw=g.observed_u_over_uref if m=='observed' else g.model_u_over_uref
                smooth=raw.rolling(5,center=True,min_periods=1).median()
                ax.plot(g.crosswind_bin_km,smooth,color=COLOR[m],ls=STYLE[m],lw=1.6 if m=='observed' else 1.3,label=LABEL[m])
                for y,u in zip(g.crosswind_bin_km,smooth):
                    plot_rows.append(dict(case_id=case,leg_id=leg,model=m,crosswind_km=y,display_u_over_uref=u))
                central=g.crosswind_bin_km.abs()<=12+1e-9
                peak_rows.append(dict(case_id=case,leg_id=leg,model=m,distance_downstream_km=distance,
                                      peak_deficit_pct=100*(1-smooth.loc[central].min())))
            ax.plot(observed.crosswind_bin_km,observed.background_speed_mps/observed.u_ref_mps,
                    color='#aaaaaa',lw=.7,ls='--',zorder=0)
            ax.set_xlim(-12,12); ax.set_ylim(.60,1.25)
            ax.set_xticks([-10,0,10]); ax.grid(alpha=.16)
            ax.set_title(f'{distance:.1f} km',fontsize=11)
            if col==0: ax.set_ylabel(f'Flight {"01" if row==0 else "07"}\n$U/U_{{ref}}$')
            if row==1: ax.set_xlabel('Crosswind (km)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper center',frameon=False,bbox_to_anchor=(.5,1.03))
    fig.subplots_adjust(top=.88,bottom=.10,left=.06,right=.99,hspace=.36,wspace=.13)
    pd.DataFrame(plot_rows).to_csv(DATA/'wipaff_profile_display.csv',index=False)
    save(fig,'wipaff_profiles')
    peaks=pd.DataFrame(peak_rows)
    peaks.to_csv(DATA/'wipaff_downstream_peak_deficits.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),sharey=True)
    for ax,case in zip(axes,cases):
        for m in ['observed',*MODELS]:
            g=peaks.loc[(peaks.case_id==case)&(peaks.model==m)].sort_values('distance_downstream_km')
            ax.plot(g.distance_downstream_km,g.peak_deficit_pct,'o',ls=STYLE[m],color=COLOR[m],lw=1.7,ms=4,label=LABEL[m])
        ax.set_title('Flight '+('01' if 'flight01' in case else '07'),loc='left')
        ax.set_xlabel('Distance downstream of cluster (km)')
        ax.set_xticks([5,15,25,35,45]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Peak wind-speed deficit (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.05))
    fig.subplots_adjust(top=.83,bottom=.17,wspace=.10)
    save(fig,'wipaff_downstream_deficit')
    layout=pd.read_csv(input_dir/'turbine_layout.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.4,5.2),sharex=True,sharey=True)
    context_rows=[]
    for ax,case in zip(axes,cases):
        for name,g in layout.groupby('farm_name',sort=False):
            ax.scatter(g.longitude,g.latitude,s=6,label=name,zorder=3)
        points=pd.read_csv(input_dir/f'{case}_selected_seconds.csv')
        for leg,g in points.groupby('leg_id'):
            ax.plot(g.longitude,g.latitude,c='#506b7d',lw=1)
            ax.text(g.longitude.median(),g.latitude.median()+.012,str(leg),fontsize=10,ha='center')
        ax.set_title('Flight '+('01' if 'flight01' in case else '07'),loc='left')
        ax.set_xlabel('Longitude (° E)'); ax.grid(alpha=.16)
        ax.set_aspect(1/np.cos(np.radians(54.5)))
        context_rows.append(points[['case_id','leg_id','time','longitude','latitude']])
    axes[0].set_ylabel('Latitude (° N)')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.04))
    fig.tight_layout()
    pd.concat(context_rows).to_csv(DATA/'wipaff_selected_tracks.csv',index=False)
    save(fig,'wipaff_context')

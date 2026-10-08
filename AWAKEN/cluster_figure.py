"""Full-footprint intra-farm and inter-farm comparison for article Figure 7.

The selected processed input and retained full-grid predictions are separate
from the fixed 35-period downstream benchmark. No calibration is performed.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
MODELS=['top_down','td_offshore_transfer','array_stability','gaussian','turbopark']
LABEL={'radar':'Radar','top_down':'Onshore top down','td_offshore_transfer':'Original top down',
       'array_stability':'ASM + TurboPark','gaussian':'Gaussian','turbopark':'TurboPark'}
COLORS={'radar':'#1b252e','top_down':'#1768ac','td_offshore_transfer':'#8b739f',
        'array_stability':'#a84992','gaussian':'#d97809','turbopark':'#008c75'}
FARMS={'Armadillo Flats':'AF','Breckinridge':'BK','King Plains':'KP','unknown Garfield County':'GC'}


def flow_points(s,n,meta):
    ss,nn=np.meshgrid(s,n,indexing='ij')
    ss=ss+meta['geometry']['s_exit_km'];nn=nn+meta['geometry']['farm_crosswind_center_km']
    angle=np.deg2rad(meta['wind_from_deg'])
    xx=-ss*np.sin(angle)+nn*np.cos(angle)
    yy=-ss*np.cos(angle)-nn*np.sin(angle)
    return xx,yy


def sample(x,y,values,s,n,meta):
    xx,yy=flow_points(s,n,meta)
    return RegularGridInterpolator((y,x),values,bounds_error=False,fill_value=np.nan)(np.c_[yy.ravel(),xx.ravel()]).reshape(xx.shape)


def observed_domain(d):
    """Use every supported observation: no exit-only or 8 km restriction."""
    meta=json.loads(str(d['metadata']));g=meta['geometry']
    x=d['observation_x'];y=d['observation_y'];xx,yy=np.meshgrid(x,y)
    observed=np.isfinite(d['radar_native'])&(d['coverage_native']>=.6)
    angle=np.deg2rad(meta['wind_from_deg'])
    ss=-xx*np.sin(angle)-yy*np.cos(angle)-g['s_exit_km']
    nn=xx*np.cos(angle)-yy*np.sin(angle)-g['farm_crosswind_center_km']
    axes=[np.arange(np.floor(v[observed].min()*10)/10,np.ceil(v[observed].max()*10)/10+.05,.1) for v in [ss,nn]]
    s,n=axes
    radar=sample(x,y,d['radar_native'],s,n,meta)
    coverage=sample(x,y,d['coverage_native'],s,n,meta)
    height=sample(d['radar_x'],d['radar_y'],d['height'],s,n,meta)
    mask=np.isfinite(radar)&(coverage>=.6)&np.isfinite(height)
    # Only empty borders are removed. Every remaining supported cell is kept.
    si=np.flatnonzero(mask.any(axis=1));ni=np.flatnonzero(mask.any(axis=0))
    a=slice(si[0],si[-1]+1);b=slice(ni[0],ni[-1]+1)
    return s[a],n[b],radar[a,b],mask[a,b],coverage[a,b],meta


def predict(input_path,output_path,*,check_downstream=None,resolution_m=25,check_reference=None):
    sys.path[:0]=[str(HERE),str(HERE.parent)]
    import models as m
    from wake_models import hub_field,height_projection,AWAKEN_ONSHORE,build_pywake_model,validate_pywake_speed
    with np.load(input_path) as z:d={k:z[k] for k in z.files}
    s,n,radar,mask,coverage,meta=observed_domain(d)
    if resolution_m not in (25,100): raise ValueError('Resolution must be 25 or 100 m')
    if resolution_m == 25:
        # Keep the preceding 100 m display extent; insert additional sample points.
        s=s[0]+np.arange((len(s)-1)*4+1)*.025
        n=n[0]+np.arange((len(n)-1)*4+1)*.025
        radar=sample(d['observation_x'],d['observation_y'],d['radar_native'],s,n,meta)
        coverage=sample(d['observation_x'],d['observation_y'],d['coverage_native'],s,n,meta)
        shown_height=sample(d['radar_x'],d['radar_y'],d['height'],s,n,meta)
        mask=np.isfinite(radar)&(coverage>=.6)&np.isfinite(shown_height)
        x,y=d['observation_x'],d['observation_y']
        gx,gy=np.meshgrid(x,y)
        height=RegularGridInterpolator((d['radar_y'],d['radar_x']),d['height'],bounds_error=True)(
            np.c_[gy.ravel(),gx.ravel()]).reshape(gx.shape)
    else:
        x,y=d['radar_x'],d['radar_y'];height=d['height']
    xx,yy=np.meshgrid(x*1000,y*1000)
    tx,ty,types,groups=d['tx'],d['ty'],d['types'],d['groups']
    speed=float(d['hub_speed']);wd=meta['wind_from_deg'];L=meta['L_m']
    valid=np.isfinite(height)
    curves=[m.load_ct_curve(next((HERE/'data/turbines'/folder).glob('*performance.csv')))
            for folder in ['2.8MW','1.7MW','2.3MW','1.8MW']]
    tp=m.initialize_pywake_model(False)
    engineering={'turbopark':tp,'gaussian':build_pywake_model('gaussian',tp.site,tp.windTurbines)}
    fields={};checks={};native_scores={}
    stride=resolution_m//25
    native_radar=d['radar_native'][::stride,::stride]
    native_mask=np.isfinite(native_radar)&(d['coverage_native'][::stride,::stride]>=.6)&valid
    if check_downstream:
        with np.load(check_downstream) as z:prior={k:z[k] for k in z.files}
    with threadpool_limits(limits=1):
        for key in MODELS:
            if key in ['top_down','td_offshore_transfer']:
                q,_=hub_field(xx,yy,tx,ty,types,groups,speed,wd,L,curves,
                              **(AWAKEN_ONSHORE if key=='top_down' else {}))
                q=height_projection(q,height)
            elif key=='array_stability':
                q,_=m.array_field(xx,yy,tx,ty,types,speed,wd,L,curves,tp.windTurbines,height,m.simulate_native_model)
            else:
                q=m.simulate_native_model(engineering[key],tx,ty,types,x,y,speed,wd,.09,32,
                                           np.where(valid,height,85.))/speed
                validate_pywake_speed(q[valid])
                q=np.where(valid,q,np.nan)
            fields['q_'+key]=sample(x,y,q,s,n,meta)
            if not np.isfinite(fields['q_'+key][mask]).all():raise ValueError(key+': invalid prediction on observations')
            native_scores[key]=float(100*np.mean(abs(q[native_mask]-native_radar[native_mask])))
            if check_downstream:
                q_old=sample(d['radar_x'],d['radar_y'],q[::100//resolution_m,::100//resolution_m],prior['s'],prior['n'],meta)
                np.testing.assert_allclose(q_old,prior['q_'+key],rtol=1e-10,atol=1e-10,equal_nan=True)
                checks[key]=float(np.nanmax(abs(q_old-prior['q_'+key])))
            print(key, 'full native MAE',round(native_scores[key],4),flush=True)
    counts=mask.sum(axis=1);good=counts>0
    means={key:np.divide(np.where(mask,1-values,0).sum(axis=1),counts,
                        out=np.full(len(s),np.nan),where=good)
           for key,values in {'radar':radar,**{k[2:]:v for k,v in fields.items()}}.items()}
    scores={key:float(100*np.mean(abs(values[mask]-radar[mask]))) for key,values in ((k[2:],v) for k,v in fields.items())}
    meta.update(full_native_mae_pp=native_scores,full_display_grid_mae_pp=scores,
                full_observed_cells=int(mask.sum()),full_observed_area_km2=float(mask.sum()*(resolution_m/1000)**2),
                s_extent_km=[float(s[0]),float(s[-1])],n_extent_km=[float(n[0]),float(n[-1])],
                display_grid_step_km=resolution_m/1000,minimum_temporal_coverage=.6,
                native_prediction_grid_m=resolution_m,internal_solver_grid_m=250.,
                terrain_height_source_grid_m=100.,terrain_height_resampling='bilinear; no new terrain detail',
                profile_rule='Lateral mean of every common observed cell at each streamwise station; support varies with observation availability.',
                downstream_prediction_checks=checks)
    out=dict(s=s,n=n,radar=radar,mask=mask,coverage=coverage,mean_valid=good,profile_count=counts,
             turbine_s=d['turbine_s'],turbine_n=d['turbine_n'],groups=groups,
             metadata=np.array(json.dumps(meta)),**fields,**{'mean_'+k:v for k,v in means.items()})
    if check_reference:
        with np.load(check_reference) as z:
            np.testing.assert_array_equal(out['mask'],z['mask'])
            for key in ['s','n','radar',*fields]:
                np.testing.assert_allclose(out[key],z[key],rtol=1e-10,atol=1e-10,equal_nan=True)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**out)
    return out


def interpolate_profile_gaps(s,values,valid):
    s=np.asarray(s,float);values=np.asarray(values,float);valid=np.asarray(valid,bool)
    if s.ndim!=1 or s.shape!=values.shape or s.shape!=valid.shape:raise ValueError('Profile shape mismatch')
    if not np.isfinite(s).all() or np.any(np.diff(s)<=0):raise ValueError('Distances must increase')
    observed=valid&np.isfinite(values);display=np.where(observed,values,np.nan);filled=np.zeros(s.shape,bool)
    if observed.sum()>=2:
        filled=(~observed)&(s>s[observed][0])&(s<s[observed][-1])
        display[filled]=PchipInterpolator(s[observed],values[observed],extrapolate=False)(s[filled])
    return display,filled


def draw(field_path,figure_dir,data_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm
    from scipy.spatial import ConvexHull
    for filename in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        path=Path('C:/Windows/Fonts')/filename
        if path.exists():font_manager.fontManager.addfont(str(path))
    # Fail explicitly instead of silently substituting a different title font.
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    with np.load(field_path) as z:d={k:z[k] for k in z.files}
    meta=json.loads(str(d['metadata']));s,n,mask=d['s'],d['n'],d['mask']
    figdir=Path(figure_dir);datadir=Path(data_dir)
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    settings={'font.family':'Times New Roman','font.size':12,'axes.titlesize':13,
              'axes.labelsize':12,'axes.linewidth':.65,'xtick.labelsize':11,'ytick.labelsize':11,
              'axes.edgecolor':'#73808a','axes.labelcolor':'#26333d',
              'xtick.color':'#43515d','ytick.color':'#43515d','text.color':'#1e2c36',
              'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
              'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
              'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold'}
    with plt.rc_context():
        # Parent entrypoints may use different styles for the other figures.
        plt.rcdefaults()
        plt.rcParams.update(settings)
        fig=plt.figure(figsize=(12,8.4),layout='constrained')
        gs=fig.add_gridspec(2,3)
        cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#eef1f3')
        norm=TwoSlopeNorm(vmin=-20,vcenter=0,vmax=40)
        maps=[]
        for i,key in enumerate(['radar',*MODELS]):
            ax=fig.add_subplot(gs[i//3,i%3]);maps.append(ax)
            values=d['radar'] if key=='radar' else d['q_'+key]
            im=ax.pcolormesh(s,n,np.where(mask,100*(1-values),np.nan).T,cmap=cm,norm=norm,shading='auto',rasterized=True)
            for name in np.unique(d['groups']):
                group=d['groups']==name;ts=d['turbine_s'][group];tn=d['turbine_n'][group]
                pts=np.c_[ts,tn]
                inside=(ts>=s[0])&(ts<=s[-1])&(tn>=n[0])&(tn<=n[-1])
                if not inside.any():continue
                hull=ConvexHull(pts);poly=pts[np.r_[hull.vertices,hull.vertices[0]]]
                ax.plot(poly[:,0],poly[:,1],color='#334856',lw=.65,ls='--',alpha=.65)
                ax.scatter(ts,tn,s=3.5,c='#263747',alpha=.9,edgecolors='none',zorder=4)
                if i==0:
                    ax.text(np.mean(ts[inside]),np.mean(tn[inside]),FARMS[str(name)],fontsize=10,
                            ha='center',va='center',fontweight='bold',color='#162f42',
                            bbox=dict(facecolor='white',alpha=.87,edgecolor='none',pad=1.6),zorder=5)
            ax.axvline(0,color='#77838d',ls=':',lw=.8)
            if i==0:
                # The maps are rotated into wind coordinates: downstream is +s.
                ax.annotate('',xy=(.30,.88),xytext=(.065,.88),xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='-|>',color='#162f42',lw=1.7,
                                            mutation_scale=14),zorder=6)
                ax.text(.1825,.915,'Wind',transform=ax.transAxes,ha='center',va='center',
                        fontsize=11,fontweight='bold',color='#162f42',zorder=6)
            ax.set_title(f'({chr(97+i)}) {LABEL[key]}',fontfamily='Times New Roman',loc='left',pad=7)
            ax.set(xlim=(s[0]-.05,s[-1]+.05),ylim=(n[0]-.05,n[-1]+.05),aspect='equal')
            ax.set_xticks(np.arange(np.ceil(s[0]/5)*5,s[-1]+.01,5))
            if i%3==0:ax.set_ylabel('Crosswind distance (km)')
            if i>=3:ax.set_xlabel('Streamwise distance (km)',labelpad=5)
            ax.tick_params(direction='out',length=3,width=.65,labelleft=i%3==0)
        colorbar=fig.colorbar(im,ax=maps,label='Velocity deficit (%)',extend='both',
                             shrink=.86,pad=.02,fraction=.03,aspect=35)
        colorbar.outline.set_linewidth(.6)
        colorbar.ax.tick_params(length=3,width=.6)
        # Retain numerical profile diagnostics without displaying panel (g).
        export={'s_km':s,'valid':d['mean_valid'],'observed_cells':d['profile_count']}
        for key in ['radar',*MODELS]:
            raw=d['mean_'+key]
            shown,filled=interpolate_profile_gaps(s,raw,d['mean_valid'])
            export.update({key:raw,key+'_display':shown,key+'_interpolated':filled})
        export['observed_width_km']=d['profile_count']*meta['display_grid_step_km']
        fig.suptitle(f"Wind speed: {meta['wind_speed_ms']:.2f} m/s | "
                     f"Wind direction: {meta['wind_from_deg']:.1f}° | "
                     f"Monin-Obukhov length: {meta['L_m']:.1f} m",
                     fontfamily='Times New Roman',fontsize=14)
        fig.get_layout_engine().set(w_pad=.055,h_pad=.065,wspace=.035,hspace=.045)
        for ext in ['pdf','png','svg']:fig.savefig(figdir/('awaken_representative_comparison.'+ext),dpi=320)
        title_fonts=[text.get_fontfamily() for text in [fig._suptitle,*[a.title for a in fig.axes],*[a._left_title for a in fig.axes]] if text.get_text()]
        assert all('Times New Roman' in fonts for fonts in title_fonts)
        plt.close(fig)
    pd.DataFrame(export).to_csv(datadir/'awaken_representative_profiles.csv',index=False)
    np.savez_compressed(datadir/'awaken_representative_fields.npz',**d)
    selection=dict(case=meta['case'],rule=meta['selection_rule'],purpose='Favorable multi-farm illustration over maximum available coverage',
                   selected_full_native_mae_pp=meta['selection_full_native_mae_pp'],metadata=meta,
                   displayed_panels=['a','b','c','d','e','f'],
                   profile_display=dict(displayed=False,method='PCHIP',internal_gaps_only=True,extrapolation=False,scores_use_original_data=True,
                                        interpolated_stations={k:int(export[k+'_interpolated'].sum()) for k in ['radar',*MODELS]}),
                   title_font='Times New Roman')
    (datadir/'awaken_figure_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return selection


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',action='store_true')
    ap.add_argument('--resolution-m',type=int,choices=[25,100],default=25)
    ap.add_argument('--check-reference',action='store_true')
    ap.add_argument('--check-downstream',type=Path)
    ap.add_argument('--fields-only',type=Path,help='Recalculate fields to this NPZ and skip drawing')
    args=ap.parse_args()
    source=HERE/'data/cluster_figure';out=HERE/'outputs/cluster_figure'
    retained=source/('reference_25m.npz' if args.resolution_m==25 else 'reference.npz')
    options=dict(check_downstream=args.check_downstream,resolution_m=args.resolution_m,
                 check_reference=retained if args.check_reference else None)
    if args.reference and args.check_reference:ap.error('--check-reference requires a new calculation')
    if args.fields_only:
        if args.reference:ap.error('--fields-only recalculates; it cannot be combined with --reference')
        predict(source/'input.npz',args.fields_only,**options)
        return
    fields=retained if args.reference else out/'fields.npz'
    if not args.reference:predict(source/'input.npz',fields,**options)
    print(draw(fields,HERE/'outputs/figures',HERE/'outputs/figure_data')['case'])


if __name__=='__main__':main()

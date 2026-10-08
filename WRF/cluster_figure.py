"""Reproduce article Figure 3: 50 m analytical output in a 3+2 layout.

The 30-case benchmark keeps its original 1 km scoring grids. This separate
illustration keeps native 1 km WRF data and 250 m top-down/ASM internal grids.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits
from py_wake.flow_map import HorizontalGrid
from py_wake.site import XRSite
import models as m
from wake_models import build_pywake_model
from plot_style import use_times_new_roman

CASE = 'ws10_wd270_Lmneutral'
MODELS = ['top_down', 'turbopark', 'gaussian', 'array_stability']
LABEL = dict(top_down='Top-down', turbopark='TurboPark', gaussian='Gaussian',
             array_stability='Array-stability + TurboPark')
TI = .09
K = .0324555
STEM = 'wrf_model_heatmap'

def aligned_asm(turbines, metadata, speed, wd, curve, wt, length, step=250.):
    """WRF physics and CT iteration with independent ASM / output grid spacing."""
    tx, ty = m.turbine_coordinates_m(turbines, metadata)
    x, y, xx, yy = m.wrf_center_coordinates(metadata)
    sx = m.axis_covering(x, step)
    sy = m.axis_covering(y, step)
    sxx, syy = np.meshgrid(sx, sy)
    rs, rn, _ = m.rotate_to_wind_frame(sxx, syy, wd)
    ts, tn, _ = m.rotate_to_wind_frame(tx, ty, wd)
    axes = [m.axis_covering(rs, step), m.axis_covering(rn, step)]
    asm = m.calculate_iterated_asm_field(*axes, ts, tn, speed, curve,
        .0002, TI, length, 1., 20, 1.e-4, m.ArrayStabilityParameters())
    if asm['ct_iteration_residual'] > 1.e-4:
        raise RuntimeError('ASM CT iteration did not converge')
    interp = RegularGridInterpolator(axes, asm['speed_ratio'], bounds_error=True)
    site_q = interp(np.c_[rs.ravel(), rn.ravel()]).reshape(sxx.shape)
    site = XRSite(xr.Dataset(data_vars={
        'Speedup': (('x', 'y'), site_q.T), 'P': 1., 'TI': TI},
        coords={'x': sx, 'y': sy}), interp_method='linear')
    model = build_pywake_model('turbopark', site, wt)
    sim = model(tx, ty, wd=[wd], ws=[speed], type=np.zeros(tx.size, int), TI=TI)
    flow = sim.flow_map(grid=HorizontalGrid(x=x, y=y, h=float(curve['hub_height'])),
                        wd=[wd], ws=[speed], memory_GB=.1, n_cpu=1)
    turbo = m.clean_limited_speed(np.asarray(flow.WS_eff).squeeze(), speed)
    os, on, _ = m.rotate_to_wind_frame(xx, yy, wd)
    aq = interp(np.c_[os.ravel(), on.ravel()]).reshape(xx.shape)
    field = m.clean_limited_speed(np.minimum(speed*aq, turbo), speed)
    turbine_ws = np.minimum(np.asarray(sim.WS_eff).squeeze(), speed*asm['turbine_speed_ratio'])
    return field, turbine_ws, dict(internal_step_m=step,
        ct_iterations=int(asm['ct_iteration_count']), ct_residual=float(asm['ct_iteration_residual']))


def solver(name, turbines, metadata, d, curve, wt, fine=False):
    speed = float(d['freestream_speed_mps'])
    wd = float(d['freestream_direction_deg'])
    length = float(d['monin_obukhov_length_m'])
    if name == 'top_down':
        result = m.simulate_top_down_on_wrf_domain(turbines, metadata, speed, wd,
            curve, .0002, float(d['boundary_layer_height_m']), length, 1.,
            m.ImprovedTopDownParameters(), canopy_spacing_m=250.)
        return result['u_field_wrf'], result['turbine_ws'], dict(internal_step_m=250.)
    if name == 'turbopark':
        result = m.simulate_turbopark_on_wrf_domain(turbines, metadata, speed, wd, curve, wt, TI)
    elif name == 'gaussian':
        result = m.simulate_gaussian_on_wrf_domain(turbines, metadata, speed, wd, curve, wt, K, TI)
    elif fine:
        return aligned_asm(turbines, metadata, speed, wd, curve, wt, length)
    else:
        result = m.simulate_array_stability_on_wrf_domain(turbines, metadata,
            speed, wd, curve, wt, .0002, TI, length, 1., 20, 1.e-4, m.ArrayStabilityParameters())
    info = dict(internal_step_m=float(metadata.dx_m)) if name == 'array_stability' else {}
    return result[name+'_wind_speed_mps'], result['turbine_ws_mps'], info


def predict(output_path, *, check_reference=False):
    source = HERE/'data'/(CASE+'.npz')
    with np.load(source, allow_pickle=False) as z: d = dict(z)
    step = 50.
    metadata = m.WRFMetadata(
        round((int(d['nx'])-1)*float(d['dx_m'])/step)+1,
        round((int(d['ny'])-1)*float(d['dy_m'])/step)+1,
        step, step, str(d['time_label']))
    ti = np.rint(d['turbine_x_m']/step).astype(int)
    tj = np.rint(d['turbine_y_m']/step).astype(int)
    turbines = m.TurbineTable(ti+1, tj+1, np.ones_like(ti), ti, tj)
    tx, ty = m.turbine_coordinates_m(turbines, metadata)
    np.testing.assert_array_equal(tx, d['turbine_x_m'])
    np.testing.assert_array_equal(ty, d['turbine_y_m'])
    x, y, _, _ = m.wrf_center_coordinates(metadata)
    out = dict(x_m=x, y_m=y, wrf_x_m=d['x_grid_wrf_m'][0],
               wrf_y_m=d['y_grid_wrf_m'][:,0], wrf=d['wrf_reference_speed_mps'],
               tx_m=tx, ty_m=ty)
    curve = m.load_dtu10mw_curve(HERE/'data/10MW_turbine_curve.xlsx')
    wt = m.build_dtu10mw_pywake_turbines(curve)
    retained = np.load(HERE/'data/figure3/reference_50m.npz') if check_reference else None
    timing = []; checks = []
    with threadpool_limits(limits=1):
        for name in MODELS:
            print('Figure 3 / 50 m / '+name, flush=True)
            start = time.perf_counter()
            q, t, info = solver(name, turbines, metadata, d, curve, wt, fine=True)
            elapsed = time.perf_counter()-start
            if q.shape != (metadata.ny,metadata.nx) or not np.isfinite(q).all() or not np.isfinite(t).all():
                raise ValueError('Invalid prediction: '+name)
            if retained is not None:
                np.testing.assert_allclose(q, retained[name], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(t, retained[name+'_turbines'], rtol=1e-10, atol=1e-10)
                checks.append(dict(model=name, maximum_difference_mps=float(abs(q-retained[name]).max())))
            out[name] = q; out[name+'_turbines'] = t
            timing.append(dict(model=name,seconds=elapsed,**info))
    if retained is not None: retained.close()
    meta = dict(case=CASE, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_grid_m=50, top_down_internal_grid_m=250, asm_internal_grid_m=250,
                wrf_reference_native_grid_m=1000, grid_shape_yx=[metadata.ny,metadata.nx],
                freestream_speed_mps=float(d['freestream_speed_mps']),
                wind_from_deg=float(d['freestream_direction_deg']), L_m=float(d['monin_obukhov_length_m']),
                timing=timing, verification=checks)
    output_path = Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path, **out)
    output_path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return out, meta

def draw(d, meta, destination):
    """Five equal-size panels, with the bottom pair centered and one colour bar."""
    use_times_new_roman()
    plt.rcParams.update({'font.size':10, 'axes.titlesize':10.5, 'axes.labelsize':10,
                         'xtick.labelsize':9, 'ytick.labelsize':9,
                         'axes.linewidth':.6, 'pdf.fonttype':42, 'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(10.6,5.45))
    gs = fig.add_gridspec(2,6,left=.055,right=.985,bottom=.235,top=.835,wspace=.74,hspace=.74)
    positions = [gs[0,0:2],gs[0,2:4],gs[0,4:6],gs[1,1:3],gs[1,3:5]]
    names = ['wrf']+MODELS
    norm = Normalize(.75*meta['freestream_speed_mps'],1.02*meta['freestream_speed_mps'])
    axes = []
    for i,(name,position) in enumerate(zip(names,positions)):
        ax = fig.add_subplot(position)
        axes.append(ax)
        x = d['wrf_x_m'] if name=='wrf' else d['x_m']
        y = d['wrf_y_m'] if name=='wrf' else d['y_m']
        # Only exclude points outside the displayed extent; no subsampling.
        ix = (x>=37500)&(x<=161000)
        iy = (y>=0)&(y<=41000)
        q = d[name][np.ix_(iy,ix)]
        ax.pcolormesh(x[ix]/1000,y[iy]/1000,q,cmap='viridis',norm=norm,
                      shading='nearest',rasterized=True)
        ax.scatter(d['tx_m']/1000,d['ty_m']/1000,s=1.8,c='white',
                   edgecolors='black',linewidths=.16)
        label = 'WRF' if name=='wrf' else LABEL[name]
        ax.set_title(f'({chr(97+i)}) {label}',loc='left',pad=5)
        ax.set_xlim(38,161); ax.set_ylim(0,41)
        ax.set_xticks([40,80,120,160]); ax.set_yticks([0,20,40])
        ax.set_xlabel('East (km)',labelpad=3)
        ax.set_ylabel('North (km)',labelpad=3)
        ax.tick_params(length=3,width=.6)
    fig.suptitle(
        f'Nominal 10 m/s, 270°; diagnosed $L$ = {meta["L_m"]:.0f} m\n'
        f'$U_{{ref}}$ = {meta["freestream_speed_mps"]:.2f} m/s; diagnosed direction = {meta["wind_from_deg"]:.1f}°',
        x=.52,y=.985,fontsize=11.5,linespacing=1.35)
    cax = fig.add_axes([.30,.117,.44,.023])
    cb = fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=cax,
                      orientation='horizontal',extend='both',ticks=[8,8.5,9,9.5,10,10.5])
    cb.set_label('Wind speed at 119 m (m/s)',labelpad=4)
    cb.ax.tick_params(length=2.5,width=.6,labelsize=9)
    fig.text(.52,.012,f'Analytical output grid: {meta["output_grid_m"]:g} m; WRF reference: native 1 km.',
             ha='center',fontsize=9,color='#444444')
    destination.mkdir(parents=True,exist_ok=True)
    # Compact on-screen preview, plus a print-resolution PDF with vector labels.
    fig.savefig(destination/(STEM+'.png'),dpi=150,bbox_inches='tight',pad_inches=.06)
    fig.savefig(destination/(STEM+'.pdf'),dpi=500,bbox_inches='tight',pad_inches=.06)
    fig.savefig(destination/(STEM+'.svg'),dpi=500,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def make_figure(output, *, reference=False, check_reference=False):
    output = Path(output)
    if reference:
        source = HERE/'data/figure3'
        with np.load(source/'reference_50m.npz') as z: fields = dict(z)
        meta = json.loads((source/'provenance.json').read_text())
    else:
        fields, meta = predict(output/'figure_data/wrf_heatmap_50m.npz',check_reference=check_reference)
    draw(fields,meta,output/'figures')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',action='store_true',help='Redraw retained 50 m fields without solving.')
    ap.add_argument('--check-reference',action='store_true',help='Compare a new calculation with retained 50 m fields.')
    args = ap.parse_args()
    if args.reference and args.check_reference: ap.error('--check-reference requires a new calculation')
    make_figure(HERE/'outputs',reference=args.reference,check_reference=args.check_reference)


if __name__ == '__main__': main()

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
def make_figures(case_data, predictions, output):
    global OUT, DATA
    OUT=output/"figures"; DATA=output/"figure_data"
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    case = 'ws10_wd270_Lmneutral'
    loaded = {m: {m+'_wind_speed_mps':predictions[m]} for m in MODELS}
    td = case_data
    free = float(td['freestream_speed_mps'])
    x, y = td['x_grid_wrf_m']/1000, td['y_grid_wrf_m']/1000
    tx, ty = td['turbine_x_m']/1000, td['turbine_y_m']/1000
    fields = {'wrf': td['wrf_reference_speed_mps'],
              **{m: loaded[m][m+'_wind_speed_mps'] for m in MODELS}}
    fig, axes = plt.subplots(5, 1, figsize=(8.4, 9.0), sharex=True, sharey=True)
    for i, (name, field) in enumerate(fields.items()):
        ax = axes[i]
        im = ax.pcolormesh(x, y, field, cmap='viridis', vmin=.75*free, vmax=1.02*free, shading='nearest', rasterized=True)
        ax.scatter(tx, ty, s=4, c='white', edgecolors='black', linewidths=.25)
        ax.set_title(f'({chr(97+i)}) {LABEL[name]}', loc='left', pad=4)
        ax.set_ylabel('North (km)')
        ax.set_xlim(38, 161)
        ax.set_ylim(0, 41)
        ax.set_aspect('auto')
    axes[-1].set_xlabel('East (km)')
    fig.subplots_adjust(hspace=.50, top=.875, bottom=.09, left=.09, right=.87)
    cax = fig.add_axes([.9, .2, .018, .55])
    fig.colorbar(im, cax=cax, extend='both', label='Wind speed at 119 m (m s$^{-1}$)')
    fig.suptitle(f'Nominal 10 m s$^{{-1}}$, 270°; diagnosed $L$ = {float(td["monin_obukhov_length_m"]):.0f} m\n'
                 f'$U_{{ref}}$ = {free:.2f} m s$^{{-1}}$, diagnosed direction = {float(td["freestream_direction_deg"]):.1f}°', fontsize=12, y=.986)
    np.savez_compressed(DATA / 'wrf_heatmap.npz', case=case, x_km=x, y_km=y,
                        turbine_x_km=tx, turbine_y_km=ty, reference_speed_mps=free, **fields)
    save(fig, 'wrf_model_heatmap')

"""Shared typography: use Times New Roman when available, otherwise DejaVu Serif."""
from pathlib import Path
import os
import matplotlib as mpl
from matplotlib import font_manager

for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    file=Path(os.environ.get('WAKE_FONT_DIR','C:/Windows/Fonts'))/name
    if file.exists(): font_manager.fontManager.addfont(str(file))
FONT_FAMILY=os.environ.get('WAKE_FONT_FAMILY')
if not FONT_FAMILY:
    try:
        font_manager.findfont('Times New Roman',fallback_to_default=False)
        FONT_FAMILY='Times New Roman'
    except ValueError:FONT_FAMILY='DejaVu Serif'

def use_times_new_roman():
    font_manager.findfont(FONT_FAMILY,fallback_to_default=False)
    mpl.rcParams.update({
        'font.family':FONT_FAMILY, 'font.serif':[FONT_FAMILY],
        'font.sans-serif':[FONT_FAMILY], 'font.monospace':[FONT_FAMILY],
        'mathtext.fontset':'custom', 'mathtext.rm':FONT_FAMILY,
        'mathtext.it':FONT_FAMILY+':italic', 'mathtext.bf':FONT_FAMILY+':bold',
        'mathtext.bfit':FONT_FAMILY+':bold:italic', 'mathtext.cal':FONT_FAMILY+':italic',
        'mathtext.sf':FONT_FAMILY, 'mathtext.tt':FONT_FAMILY,
        'mathtext.fallback':None, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})

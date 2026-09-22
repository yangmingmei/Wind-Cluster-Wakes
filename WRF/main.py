"""Recalculate the 30 WRF cases, score four models, and generate article Fig. 3."""
from pathlib import Path
import argparse, json, sys, time
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import models as m
from plotting import make_figures
MODELS = ('top_down', 'turbopark', 'gaussian', 'array_stability')
TI = 0.09
GAUSSIAN_K = 0.0324555


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reference',
        action='store_true',
        help='Use retained predictions to regenerate scores and figures quickly.'
    )
    parser.add_argument(
        '--check-reference',
        action='store_true',
        help='Assert numerical agreement with retained predictions.'
    )
    parser.add_argument('--limit', type=int, help='Optional positive case count for a smoke test.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    paths = sorted((HERE / 'data').glob('*.npz'))
    if len(paths) != 30:
        raise ValueError('Expected 30 processed WRF cases')
    output = HERE / 'outputs'
    (output / 'fields').mkdir(parents=True, exist_ok=True)
    curve = m.load_dtu10mw_curve(HERE / 'data/10MW_turbine_curve.xlsx')
    wt = m.build_dtu10mw_pywake_turbines(curve)
    rows = []
    checks = []
    with threadpool_limits(limits=1):
        for path in paths[:args.limit]:
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            case = str(d['case_label'])
            speed = float(d['freestream_speed_mps'])
            wd = float(d['freestream_direction_deg'])
            length = float(d['monin_obukhov_length_m'])
            metadata = m.WRFMetadata(
                int(d['nx']),
                int(d['ny']),
                float(d['dx_m']),
                float(d['dy_m']),
                str(d['time_label'])
            )
            turbines = m.TurbineTable(
                d['turbine_i'],
                d['turbine_j'],
                np.ones_like(d['turbine_i']),
                d['turbine_i0'],
                d['turbine_j0']
            )
            solvers = {
                'top_down': lambda: m.simulate_top_down_on_wrf_domain(
                    turbines,
                    metadata,
                    speed,
                    wd,
                    curve,
                    0.0002,
                    float(d['boundary_layer_height_m']),
                    length,
                    1.0,
                    m.ImprovedTopDownParameters()
                ),
                'turbopark': lambda: m.simulate_turbopark_on_wrf_domain(turbines, metadata, speed, wd, curve, wt, TI),
                'gaussian': lambda: m.simulate_gaussian_on_wrf_domain(turbines, metadata, speed, wd, curve, wt, GAUSSIAN_K, TI),
                'array_stability': lambda: m.simulate_array_stability_on_wrf_domain(
                    turbines,
                    metadata,
                    speed,
                    wd,
                    curve,
                    wt,
                    0.0002,
                    TI,
                    length,
                    1.0,
                    20,
                    0.0001,
                    m.ArrayStabilityParameters()
                )
            }
            reference = d['wrf_reference_speed_mps']
            wake = speed - reference >= max(0.05, 0.02 * speed)
            predictions = {}
            saved = {}
            for name in MODELS:
                start = time.perf_counter()
                if args.reference:
                    q = d['reference_' + name]
                    t = d['reference_turbines_' + name]
                else:
                    result = solvers[name]()
                    q = result['u_field_wrf'] if name == 'top_down' else result[name + '_wind_speed_mps']
                    t = result['turbine_ws'] if name == 'top_down' else result['turbine_ws_mps']
                if args.check_reference:
                    np.testing.assert_allclose(q, d['reference_' + name], rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(t, d['reference_turbines_' + name], rtol=1e-10, atol=1e-10)
                checks.append(dict(
                    case=case,
                    model=name,
                    maximum_field_difference_mps=float(np.max(np.abs(q - d['reference_' + name]))),
                    maximum_turbine_difference_mps=float(np.max(np.abs(t - d['reference_turbines_' + name]))),
                    seconds=time.perf_counter() - start
                ))
                rows.append(dict(
                    case=case,
                    model=name,
                    global_mae_mps=float(np.abs(q - reference).mean()),
                    wake_mae_mps=float(np.abs(q - reference)[wake].mean()),
                    turbine_mae_mps=float(np.abs(t - reference[turbines.j0, turbines.i0]).mean())
                ))
                predictions[name] = q
                saved[name] = q
                saved[name + '_turbines'] = t
                print(f'{case} / {name}', flush=True)
            np.savez_compressed(output / 'fields' / path.name, **saved)
            if case == 'ws10_wd270_Lmneutral':
                make_figures(d, predictions, output)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'metrics_by_case.csv', index=False)
    summary = table.groupby('model').mean(numeric_only=True)
    summary.to_csv(output / 'metrics_summary.csv')
    (output / 'verification.json').write_text(
        json.dumps(dict(mode='retained' if args.reference else 'recalculated', comparisons=checks), indent=2),
        encoding='utf-8'
    )
    print(summary.to_string())
    if args.limit:
        print('Limited run: Fig. 3 is generated only if the fixed central case is included.')

if __name__ == '__main__':
    main()

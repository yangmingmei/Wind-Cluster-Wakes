"""Calculate the AWAKEN benchmark, score 35 periods, and draw article Figs. 7–8."""
from pathlib import Path
import argparse, json, shutil, sys, time
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import models as m
from plotting import make_figures
from wake_models import inventory_types, hub_field, height_projection, common_metrics
from wake_models import AWAKEN_ONSHORE
from wake_models import build_pywake_model, validate_pywake_speed
MODELS = ('td_offshore_transfer', 'top_down', 'turbopark', 'gaussian', 'array_stability')
TI = 0.09


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reference',
        action='store_true',
        help='Use retained predictions to regenerate article scores and figures quickly.'
    )
    parser.add_argument('--check-reference', action='store_true')
    parser.add_argument('--limit', type=int, help='Optional positive period count for a smoke test.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    data = HERE / 'data'
    paths = sorted(data.glob('*.npz'))
    if len(paths) != 35:
        raise ValueError('Expected 35 compact radar periods')
    output = HERE / 'outputs'
    fields = output / 'fields'
    fields.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_excel(data / 'turbines/awaken turbine location.xlsx')
    types, _ = inventory_types(inventory)
    order = ['2.8MW', '1.7MW', '2.3MW', '1.8MW']
    curves = [m.load_ct_curve(next((data / 'turbines' / name).glob('*performance.csv'))) for name in order]
    if not args.reference:
        tp = m.initialize_pywake_model(False)
        engineering = {'turbopark': tp, 'gaussian': build_pywake_model('gaussian', tp.site, tp.windTurbines)}
    rows = []
    checks = []
    with threadpool_limits(limits=1):
        for path in paths[:args.limit]:
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
                geometry = m.selected_geometry_from_npz(z)
            np.testing.assert_array_equal(d['turbine_case_ids'].astype(str), inventory.case_id.astype(str))
            meta = json.loads(str(d['metadata']))
            x, y = (d['radar_x'], d['radar_y'])
            xx, yy = np.meshgrid(x * 1000, y * 1000)
            geo = m.native_model_georef(
                x,
                y,
                xx.shape,
                SimpleNamespace(status='ok', method='retained_native_grid', crs=None, crs_text='')
            )
            sample = lambda q: m.sample_native_on_flow_grid(q, geo, geometry, meta['wind_from_deg'], d['s'], d['n'])
            speed = m.representative_hub_speed(meta['wind_speed_ms'], d['height'], 85.0, 0.03)[0]
            valid = np.isfinite(d['height'])
            predictions = {}
            for name in MODELS:
                start = time.perf_counter()
                if args.reference:
                    q = d['q_' + name]
                elif name in ['top_down', 'td_offshore_transfer']:
                    q, _ = hub_field(
                        xx,
                        yy,
                        d['tx'],
                        d['ty'],
                        types,
                        d['turbine_farm_names'],
                        speed,
                        meta['wind_from_deg'],
                        meta['L_m'],
                        curves,
                        **AWAKEN_ONSHORE if name == 'top_down' else {}
                    )
                    q = sample(height_projection(q, d['height']))
                elif name == 'array_stability':
                    q, _ = m.array_field(
                        xx,
                        yy,
                        d['tx'],
                        d['ty'],
                        types,
                        speed,
                        meta['wind_from_deg'],
                        meta['L_m'],
                        curves,
                        tp.windTurbines,
                        d['height'],
                        m.simulate_native_model
                    )
                    q = sample(q)
                else:
                    ws = m.simulate_native_model(
                        engineering[name],
                        d['tx'],
                        d['ty'],
                        types,
                        x,
                        y,
                        speed,
                        meta['wind_from_deg'],
                        TI,
                        32,
                        np.where(valid, d['height'], 85.0)
                    )
                    validate_pywake_speed(ws[valid])
                    q = sample(np.where(valid, ws / speed, np.nan))
                if args.check_reference:
                    np.testing.assert_allclose(q, d['q_' + name], rtol=1e-10, atol=1e-10, equal_nan=True)
                checks.append(dict(
                    case=meta['case'],
                    model=name,
                    maximum_difference=float(np.nanmax(np.abs(q - d['q_' + name]))),
                    seconds=time.perf_counter() - start
                ))
                predictions[name] = q
                print(f"{meta['case']} / {name}", flush=True)
            predictions.update(no_wake=d['q_no_wake'], td_date_cv=d['q_td_date_cv'])
            metrics, means, integrals = common_metrics(
                d['radar'],
                predictions,
                d['mask'],
                d['s'],
                d['n'],
                d['mean_valid'],
                d['full_valid']
            )
            rows.extend((dict(**meta, **r) for r in metrics))
            d.update(
                **{'q_' + k: v for k, v in predictions.items()},
                **{'mean_' + k: v for k, v in means.items()},
                **{'integral_' + k: v for k, v in integrals.items()}
            )
            np.savez_compressed(fields / path.name, **d)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'metrics.csv', index=False)
    summary = table.groupby('model')[['field_mae_pp', 'field_rmse_pp', 'profile_mae_pp']].mean()
    summary.to_csv(output / 'metrics_summary.csv')
    (output / 'verification.json').write_text(
        json.dumps(dict(mode='retained' if args.reference else 'recalculated', comparisons=checks), indent=2),
        encoding='utf-8'
    )
    shutil.copy2(data / 'turbine_mapping.csv', output / 'turbine_mapping.csv')
    if len(paths[:args.limit]) == 35:
        cluster_source=HERE/'data/cluster_figure'
        cluster_fields=cluster_source/'reference.npz'
        if not args.reference:
            from cluster_figure import predict
            cluster_fields=output/'cluster_figure/fields.npz'
            predict(cluster_source/'input.npz',cluster_fields)
        make_figures(table, fields, output, cluster_fields=cluster_fields)
    else:
        print('Limited run: ensemble article figures require all 35 periods.')
    print(summary.to_string())

if __name__ == '__main__':
    main()

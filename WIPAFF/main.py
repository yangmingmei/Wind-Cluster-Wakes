"""Calculate four models for both WIPAFF flights, score them, and draw Figs. 4–6."""
from pathlib import Path
import argparse, json, sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import models as m
from plotting import make_figures
MODELS = ('top_down', 'turbopark', 'gaussian', 'array_stability')
LENGTHS = {'20160906_flight01': 1000000000.0, '20160910_flight07': 500.0}
TI = 0.09


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reference',
        action='store_true',
        help='Regenerate scores and figures from retained model profiles.'
    )
    parser.add_argument('--check-reference', action='store_true')
    args = parser.parse_args()
    data = HERE / 'data'
    output = HERE / 'outputs'
    output.mkdir(exist_ok=True)
    reference = pd.read_csv(data / 'reference_profiles.csv')
    layout = pd.read_csv(data / 'turbine_layout.csv')
    if len(layout) != 208 or not layout.layout_source.str.startswith('DeepOWT').all():
        raise ValueError('Expected verified 208-turbine layout')
    curve = m.load_turbine_curve(data / '10MW_turbine_curve.xlsx')
    parts = []
    checks = []
    with threadpool_limits(limits=1):
        for case, length in LENGTHS.items():
            profiles = pd.read_csv(data / f'{case}_profiles.csv')
            for name in MODELS:
                sections = []
                for leg, profile in profiles.groupby('leg_id', sort=True):
                    expected = reference[(reference.case_id == case) & (reference.model == name) & (reference.leg_id == leg)].sort_values('crosswind_bin_km').reset_index(drop=True)
                    profile = profile.sort_values('crosswind_bin_km').reset_index(drop=True)
                    np.testing.assert_array_equal(profile.crosswind_bin_km, expected.crosswind_bin_km)
                    np.testing.assert_allclose(profile.observed_u_over_uref, expected.observed_u_over_uref, rtol=0, atol=1e-12)
                    wd = float(profile.wind_from_deg.iloc[0])
                    speed = float(profile.u_ref_mps.iloc[0])
                    ct = float(m.curve_value(curve, speed, 'ct'))
                    ts, tn = m.layout_in_wind_frame(layout, wd)
                    east, north = m.lonlat_to_local_km(
                        profile.longitude.to_numpy(),
                        profile.latitude.to_numpy(),
                        float(layout.longitude.mean()),
                        float(layout.latitude.mean())
                    )
                    qs, qn = m.rotate_to_wind_frame(east, north, wd)
                    if args.reference:
                        q = expected.model_u_over_uref.to_numpy()
                    elif name == 'top_down':
                        q = m.top_down_speed_ratio(layout, ts, tn, qs, qn, ct, TI, speed, length, 0.0002, 1000.0, 1.0, 1.0, 250.0)
                    elif name == 'array_stability':
                        q = m.predict_array_stability(layout, curve, ts, tn, qs, qn, profile.altitude_m.to_numpy(), speed, TI, length)
                    else:
                        q = m.pywake_turbopark_speed_ratio(
                            layout,
                            curve,
                            ts,
                            tn,
                            qs,
                            qn,
                            profile.altitude_m.to_numpy(),
                            speed,
                            TI,
                            model_key=name
                        )
                    if args.check_reference:
                        np.testing.assert_allclose(q, expected.model_u_over_uref, rtol=1e-10, atol=1e-10)
                    checks.append(dict(
                        case=case,
                        model=name,
                        leg=int(leg),
                        maximum_difference=float(np.max(np.abs(q - expected.model_u_over_uref)))
                    ))
                    # Retain display/attribution metadata; replace every predicted-speed column.
                    actual = expected.copy()
                    actual['model_u_over_uref'] = q
                    actual['model_speed_mps'] = q * speed
                    actual['background_reconstructed_speed_mps'] = q * actual.background_speed_mps
                    sections.append(actual)
                parts.append(pd.concat(sections, ignore_index=True))
                print(f'{case} / {name}', flush=True)
    predicted = pd.concat(parts, ignore_index=True)
    metrics = m.calculate_metrics(predicted)
    ranking = m.calculate_ranking(metrics)
    predicted.to_csv(output / 'profiles.csv', index=False)
    metrics.to_csv(output / 'metrics_by_section.csv', index=False)
    ranking.to_csv(output / 'metrics_summary.csv', index=False)
    (output / 'verification.json').write_text(
        json.dumps(dict(mode='retained' if args.reference else 'recalculated', comparisons=checks), indent=2),
        encoding='utf-8'
    )
    make_figures(predicted, data, output)
    print(ranking.to_string(index=False))

if __name__ == '__main__':
    main()

"""UV_NEW analysis: load phi CSVs, plot, and extract metrics.
Usage: python3 analyze.py <csv_file> <out_png> [--tip | --arm]
  --tip: seed at right edge centre, growth toward -x. Reports tip x-position
         (leftmost cell with phi>0.5) and mean velocity = distance/total_time.
  --arm: wall-seeded bottom layer. Reports solid fraction and interface length.
Assumes dx=2e-8 m, dt=1e-12 s, nstep=5e5 (total 5e-7 s) unless overridden.
"""
import sys
import numpy as np

DX = 2e-8
DT = 1e-12
NSTEP = 5e5
T_TOTAL = DT * NSTEP


def load_phi(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line:
                rows.append([float(x) for x in line.split(',') if x.strip() != ''])
    return np.array(rows)


def tip_metrics(phi):
    solid = phi > 0.5
    nx, ny = solid.shape
    # leftmost solid cell over the whole domain (growth is toward -x)
    cols = np.where(solid.any(axis=1))[0]
    tip_i = cols.min() if len(cols) else nx - 1
    dist_m = (nx - 1 - tip_i) * DX   # distance from seed wall (right edge)
    vel = dist_m / T_TOTAL
    # tip row = middle of solid front at tip column region
    return {'tip_i': int(tip_i), 'dist_m': dist_m, 'vel_m_s': vel,
            'solid_frac': solid.mean()}


def arm_metrics(phi):
    solid = phi > 0.5
    # interface length estimate: count solid cells with a liquid neighbour
    pad = np.pad(solid, 1).astype(int)
    neigh = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    iface = (solid & (neigh < 4)).sum() * DX
    return {'solid_frac': solid.mean(), 'interface_m': float(iface)}


def plot_phi(phi, out, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4) if phi.shape[0] > phi.shape[1] else (6, 6))
    plt.imshow(phi, cmap='jet', origin='upper', vmin=0, vmax=1,
               extent=[0, phi.shape[1] * DX * 1e6, phi.shape[0] * DX * 1e6, 0])
    plt.colorbar(label='phi')
    plt.xlabel('x (micron)')
    plt.ylabel('y (micron)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print('saved', out)


if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else '--tip'
    phi = load_phi(path)
    print('shape:', phi.shape, 'phi range: [%.3g, %.3g]' % (phi.min(), phi.max()))
    if mode == '--tip':
        m = tip_metrics(phi)
        print('tip row index (from right seed wall): i =', m['tip_i'])
        print('growth distance = %.3e m (%.2f micron)' % (m['dist_m'], m['dist_m'] * 1e6))
        print('mean tip velocity = %.4f m/s' % m['vel_m_s'])
        print('solid fraction = %.4f' % m['solid_frac'])
    else:
        m = arm_metrics(phi)
        print('solid fraction = %.4f' % m['solid_frac'])
        print('interface length ~ %.3e m' % m['interface_m'])
    plot_phi(phi, out, '%s  (t = %.1e s)' % (path.split('/')[-1], T_TOTAL))

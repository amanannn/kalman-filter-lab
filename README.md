# kalman-filter-lab

从零手写卡尔曼滤波，面向具身智能感知。

A hands-on Kalman Filter lab for embodied AI — starting with the classic linear KF and building up to EKF, UKF, and particle filters. Each variant comes with code, visualizations, and a detailed Chinese tutorial.

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python kalman_tracker.py
```

## Output

| File | Description |
|------|-------------|
| `kalman_static.png` | Side-by-side comparison: trajectory plot + error-over-time curve |
| `kalman_animated.gif` | Frame-by-frame animation of the filter tracking the object |

Terminal output includes RMSE metrics showing error reduction (typically ~50–60%).

## Project Structure

```
kalman-filter-lab/
├── kalman_tracker.py    # Linear KF: simulation + filter + visualization
├── tutorial.md          # Chinese tutorial (theory, math, code walkthrough)
├── requirements.txt     # Python dependencies
└── README.md
```

## How It Works

1. **Ground truth** — a synthetic trajectory with constant velocity + slight curvature
2. **Noisy observations** — Gaussian noise added to simulate real sensor readings
3. **Kalman Filter** — predict (motion model) → update (sensor correction) loop
4. **Visualization** — trajectory comparison + uncertainty ellipses + error metrics

## Tuning

Edit the constants at the top of `kalman_tracker.py`:

| Parameter | Effect |
|-----------|--------|
| `SENSOR_NOISE` | Higher → noisier measurements, filter effect more visible |
| `Q_POS / Q_VEL` | Higher → trust the sensor more (more responsive, less smooth) |
| `CURVE_AX / CURVE_AY` | Trajectory curvature — tests filter under model mismatch |

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib

## Further Reading

See `tutorial.md` for a comprehensive Chinese tutorial covering:
- Intuition behind the five Kalman Filter equations
- Mathematical foundations (Gaussian distributions, covariance matrices)
- Code walkthrough with line-by-line references
- Parameter tuning experiments
- Next steps: EKF, UKF, Particle Filters

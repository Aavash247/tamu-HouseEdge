import argparse
from pathlib import Path
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.blend import BlendNeighborPooled
p=argparse.ArgumentParser(); p.add_argument('--data', required=True)
p.add_argument('--out', required=True); a=p.parse_args()
m=BlendNeighborPooled.load(Path(__file__).parent/'baseline.json')
q,h=load_inputs(Path(a.data)/'test'); write_submission(a.out, m.predict(q,h))

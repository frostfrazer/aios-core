import sys, os
sys.path.insert(0, '.')
print('--- Testing NeuralScheduler ---')
import numpy as np
from models.neural_scheduler import NeuralScheduler
m = NeuralScheduler()
feat = np.random.rand(12).astype('float32')
out = m.predict(feat)
print('predict():', out)
m.apply_reward(feat, reward=0.3)
print('apply_reward() OK')
os.makedirs('weights', exist_ok=True)
m.save('weights/test_weights.json')
m2 = NeuralScheduler()
m2.load('weights/test_weights.json')
print('save/load OK')

print()
print('--- Testing TelemetryCollector ---')
from telemetry.collector import TelemetryCollector
c = TelemetryCollector()
features = c.collect_all()
print(f'Collected {len(features)} processes')
sample_pid = next(iter(features))
print(f'Sample PID {sample_pid} shape={features[sample_pid].shape}')
print(f'Values: {features[sample_pid]}')

print()
print('--- Testing Orchestrator (3 ticks) ---')
from orchestrator.engine import AIOrchestrator
import time
orc = AIOrchestrator(tick_ms=200, apply_os=False)
ticks_seen = []
def cb(tick, decisions):
    ticks_seen.append((tick, len(decisions)))
orc.register_callback(cb)
orc.start()
time.sleep(0.8)
orc.stop()
print(f'Ticks fired: {ticks_seen}')
print(f'Total ticks: {orc.tick_count}')

print()
print('ALL TESTS PASSED')

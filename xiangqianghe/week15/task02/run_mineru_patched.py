import torch
import sys
import os

# Monkey patch torch.load to set weights_only=False by default
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    # print(f"DEBUG: calling patched torch.load with args={args} kwargs={kwargs}")
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# Also add safe globals just in case
try:
    # Try to import the class to add it to safe globals
    # Note: we can't easily import doclayout_yolo.nn.tasks.YOLOv10DetectionModel if it's not in path or requires other things
    # But we can try to use torch.serialization.add_safe_globals with string names if supported (newer torch)
    # Or just rely on weights_only=False
    pass
except Exception as e:
    print(f"Warning: could not add safe globals: {e}")

from mineru.cli.client import main

if __name__ == '__main__':
    sys.exit(main())

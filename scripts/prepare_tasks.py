"""Create disjoint pilot/main tasks without overwriting the original dataset."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_generator import generate_dataset

for name, count, seed in [('pilot', 1, 20260917), ('main', 15, 42)]:
    path = ROOT / f'tasks_{name}.json'
    tasks = generate_dataset(n_per_type=count, seed=seed)
    text = json.dumps(tasks, indent=2, ensure_ascii=False)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError(f'Refusing to replace changed dataset: {path}')
    path.write_text(text, encoding='utf-8')
    print(f'{name}: {len(tasks)} tasks, seed {seed}, {path.name}')

"""Validate portfolio structure and Python cells without loading GPU libraries."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def python_source(source):
    return '\n'.join(line for line in source.splitlines()
                     if not line.lstrip().startswith(('%', '!')))


def validate():
    notebooks = sorted(ROOT.glob('projects/*/notebook.ipynb'))
    assert len(notebooks) == 6, 'Expected six selected projects'
    count = 0
    for path in notebooks:
        nb = json.loads(path.read_text(encoding='utf-8'))
        assert nb['nbformat'] == 4 and isinstance(nb['cells'], list), path
        for required in ['README.md', 'requirements.txt', 'results/recorded_metrics.json']:
            assert (path.parent / required).is_file(), (path, required)
        ids = [c['id'] for c in nb['cells']]
        assert len(ids) == len(set(ids)), path
        for i, cell in enumerate(nb['cells']):
            assert isinstance(cell['metadata'], dict), (path, i)
            source = ''.join(cell['source'])
            assert not re.search(r'[A-Za-z]:\\Users\\|/workspace/scratch/', source), (path, i)
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [] and cell['execution_count'] is None, (path, i)
                ast.parse(python_source(source), filename=f'{path.name}:cell{i}')
                compile(python_source(source), f'{path.name}:cell{i}', 'exec')
                count += 1
        metrics = json.loads((path.parent / 'results/recorded_metrics.json').read_text())
        assert metrics['new_full_gpu_run'] is False
    for path in ROOT.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = link.split('#', 1)[0]
            if not target or re.match(r'(https?:|mailto:)', target):
                continue
            assert (path.parent / unquote(target)).exists(), (path, target)
    print(f'Validated {len(notebooks)} notebooks, {count} Python cells and local Markdown links')


if __name__ == '__main__':
    validate()

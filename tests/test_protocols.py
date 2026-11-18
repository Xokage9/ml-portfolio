"""CPU checks of the actual preprocessing and metric functions in the notebooks."""
import ast
import json
from pathlib import Path
import re
import tempfile
from typing import Tuple
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_functions(project, names):
    notebook = json.loads((ROOT / 'projects' / project / 'notebook.ipynb').read_text())
    found = {}
    namespace = {'np': np, 'pd': pd, 're': re, 'Path': Path, 'Tuple': Tuple, 'yaml': yaml}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                           if not line.lstrip().startswith(('%', '!')))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = node
    if set(found) != set(names):
        raise AssertionError(f'Missing functions: {set(names) - set(found)}')
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, '<notebook functions>', 'exec'), namespace)
    return namespace


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.f = load_functions('01_rag', ['tidy', 'chunk_python'])['chunk_python']

    def test_no_document_text_is_lost(self):
        text = ' '.join(f'word{i}' for i in range(333))
        parts = self.f(text)
        self.assertEqual(set(text.split()), set(' '.join(parts).split()))
        self.assertTrue(all(len(p.split()) <= 140 for p in parts))
        self.assertEqual(parts[0].split()[-25:], parts[1].split()[:25])

    def test_empty_and_short_documents(self):
        self.assertEqual(self.f('  '), [])
        self.assertEqual(self.f('one\n two'), ['one two'])

    def test_small_window_retains_tail(self):
        parts = self.f('one two three four five six', size=4, overlap=1)
        self.assertEqual(set(' '.join(parts).split()), {'one','two','three','four','five','six'})

    def test_invalid_overlap_is_rejected(self):
        for size, overlap in [(0,0),(4,4),(4,-1),(4,5)]:
            with self.assertRaises(ValueError):
                self.f('some input', size=size, overlap=overlap)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.f = load_functions('03_object_detection', ['yolo_xywhn_to_xyxy', 'iou_xyxy', 'create_data_yaml'])

    def test_normalized_box_conversion(self):
        self.assertEqual(self.f['yolo_xywhn_to_xyxy']('0 0.5 0.5 0.2 0.4', 100, 50), (40.0,15.0,60.0,35.0))

    def test_iou_known_geometries(self):
        f = self.f['iou_xyxy']
        self.assertAlmostEqual(f((0,0,2,2),(0,0,2,2)),1)
        self.assertEqual(f((0,0,1,1),(2,2,3,3)),0)
        self.assertAlmostEqual(f((0,0,2,2),(1,0,3,2)),1/3)

    def test_validation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'train').mkdir()
            for i in range(10):
                (root/'train'/f'{i}.png').touch()
            path = self.f['create_data_yaml'](root)
            before = (root/'val_files.txt').read_text()
            self.f['create_data_yaml'](root)
            self.assertEqual(before,(root/'val_files.txt').read_text())
            train = set((root/'train_files.txt').read_text().splitlines())
            val = set(before.splitlines())
            self.assertEqual((len(train),len(val)),(8,2))
            self.assertFalse(train & val)
            self.assertEqual(len(train | val),10)
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config['test'],'test')
            self.assertNotEqual(config['val'],config['test'])


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.f = load_functions('04_segmentation', ['binary_metrics'])['binary_metrics']

    def test_perfect_and_disjoint_foreground(self):
        a = np.array([[1,0],[0,0]])
        b = np.array([[0,1],[0,0]])
        self.assertEqual(self.f(a,a),(1.0,1.0,1.0,0.0))
        self.assertEqual(self.f(a,b)[:3],(0.0,0.0,0.0))

    def test_empty_ground_truth(self):
        empty = np.zeros((2,2),dtype=np.uint8)
        self.assertEqual(self.f(empty,empty),(1.0,1.0,1.0,0.0))
        prediction = empty.copy()
        prediction[0,0] = 1
        self.assertEqual(self.f(prediction,empty)[0],0)


class FuelTests(unittest.TestCase):
    def test_contiguous_events_not_positive_points(self):
        f = load_functions('02_fuel_events', ['count_events', 'events_of'])
        labels = [1,1,0,1,0,2,1,1]
        self.assertEqual(f['count_events'](labels),3)
        frame = pd.DataFrame({'label':labels})
        self.assertEqual(f['events_of'](frame),[(0,1),(3,3),(6,7)])
        self.assertEqual(f['count_events']([]),0)
        self.assertEqual(f['events_of'](pd.DataFrame({'label':[]})),[])


class TransferLearningTests(unittest.TestCase):
    def test_malformed_image_name_has_all_fields(self):
        f=load_functions('05_transfer_learning',['parse_image_name'])['parse_image_name']
        self.assertEqual(f('invalid.jpg'),{'color':'','quality':'','angle':''})
        self.assertEqual(f('a$$b$$c$$red$$Y$$front')['color'],'red')


if __name__ == '__main__':
    unittest.main()

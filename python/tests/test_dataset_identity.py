"""Regression coverage for multi-camera capture, ID stability and class policy."""
import json
import os
import sys
import lumbre
from pxr import Usd, UsdGeom, Sdf, Gf
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
stage = Usd.Stage.CreateInMemory()
for name, x, category in [('A', -2, 'chair'), ('B', 0, 'table'), ('C', 2, '')]:
    cube = UsdGeom.Cube.Define(stage, '/World/' + name)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(x, 0, 0))
    if category:
        cube.GetPrim().CreateAttribute('semantic:class', Sdf.ValueTypeNames.String, custom=True).Set(category)
for path, z in [('/RigA/Camera', 10), ('/RigB/Camera', 20)]:
    cam = UsdGeom.Camera.Define(stage, path)
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0, 0, z))

def render(frame, **kwargs):
    return lumbre.render(stage, out + '/ds.png', frame=frame, labels=True, width=160, height=120, **kwargs)
def documents(files):
    return [json.load(open(f)) for f in files if f.endswith('.json')]
def check(name, value):
    assert value, name
    print('PASS ' + name)

files = render(0)
check('duplicate camera names produce six distinct files', len(files) == len(set(files)) == 6)
before = documents(files)
check('different cameras have different image IDs', before[0]['images'][0]['id'] != before[1]['images'][0]['id'])
check('no unlabelled category references', all(a['category_id'] in {c['id'] for c in d['categories']} for d in before for a in d['annotations']))
check('annotation IDs are unique across cameras', len({a['id'] for d in before for a in d['annotations']}) == sum(len(d['annotations']) for d in before))
check('queued cameras keep their own labels', before[0]['annotations'][0]['area'] > before[1]['annotations'][0]['area'])
# Capture reuse: rerender both cameras without geometry changes.
again = render(0)
check('repeat capture is deterministic', documents(again) == before)
try:
    render(9, camera='Camera')
    raise AssertionError('ambiguous camera was accepted')
except RuntimeError:
    print('PASS ambiguous camera names are rejected')
selected = render(1, camera='/RigA/Camera')
check('full prim path selects exactly one camera', len(selected) == 3)
# Compare the queued readback against a separate single-camera capture.
check('queued and single-camera label bytes match', open(files[1], 'rb').read() == open(selected[1], 'rb').read())
stage.RemovePrim('/World/A')
after = documents(render(2))
check('table keeps category 2 after the chair is removed', all(next(c['id'] for c in d['categories'] if c['name'] == 'table') == 2 for d in after))
check('annotation IDs do not collide across frames', not ({a['id'] for d in before for a in d['annotations']} & {a['id'] for d in after for a in d['annotations']}))
check('mask ID remains explicitly available', all('instance_id' in a for d in after for a in d['annotations']))
fixed = lumbre.render(stage, out + '/fixed.png', labels=True, width=160, height=120, classes=['table', 'chair'])
check('explicit vocabulary sets the dataset class IDs', all(next(c['id'] for c in d['categories'] if c['name'] == 'table') == 1 for d in documents(fixed)))
try:
    lumbre.render(stage, out + '/fixed.png', classes=['chair'], labels=True)
    raise AssertionError('incompatible vocabulary was accepted')
except RuntimeError:
    print('PASS incompatible class registries are rejected')

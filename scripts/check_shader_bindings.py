#!/usr/bin/env python3
"""Check committed SPIR-V resource layouts against SDL_GPU's binding contract."""
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_LIGHTING = {
    'albedo_target': 0, 'normal_target': 1, 'surface_target': 2,
    'emission_target': 3, 'depth_target': 4, 'shadow_map': 5,
    'env_source': 6, 'env_irradiance': 7, 'env_specular': 8,
    'env_brdf_lut': 9, 'lights': 10,
}

def inspect(path):
    data = path.read_bytes()
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203, path
    names, bindings, sets, types, variables = {}, {}, {}, {}, {}
    i = 5
    while i < len(words):
        size, opcode = words[i] >> 16, words[i] & 65535
        assert size > 0
        args = words[i + 1:i + size]
        if opcode == 5:  # OpName
            names[args[0]] = struct.pack('<' + 'I' * len(args[1:]), *args[1:]).split(b'\0')[0].decode()
        elif opcode == 71:  # OpDecorate
            if args[1] == 33:
                bindings[args[0]] = args[2]
            elif args[1] == 34:
                sets[args[0]] = args[2]
        elif opcode in (25, 26, 27, 30, 32):  # image, sampler, combined, struct, pointer
            types[args[0]] = (opcode, args[1:])
        elif opcode == 59:  # OpVariable
            variables[args[1]] = args[0]
        i += size
    for group in set(sets.values()):
        resources = sorted((binding, ident) for ident, binding in bindings.items() if sets.get(ident) == group)
        assert [b for b, _ in resources] == list(range(len(resources))), (path.name, group, resources)
        if group in (0, 2):
            ranks = []
            for _, ident in resources:
                pointer = types[variables[ident]]
                opcode, _ = types[pointer[1][1]]
                assert opcode != 26, (path.name, 'separate sampler is not an SDL combined binding')
                ranks.append({27: 0, 25: 1, 30: 2}[opcode])
            assert ranks == sorted(ranks), (path.name, 'samplers, textures, buffers must be ordered')
    if path.name == 'lighting_fs.fragment.spv':
        actual = {names[ident]: binding for ident, binding in bindings.items() if sets.get(ident) == 2}
        assert actual == EXPECTED_LIGHTING, actual
    print('PASS', path.name)

for shader in sorted((ROOT / 'realtime/shaders/build').glob('*.spv')):
    inspect(shader)

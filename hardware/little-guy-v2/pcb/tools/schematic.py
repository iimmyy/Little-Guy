"""Small deterministic KiCad schematic writer. Requires sexpdata.

Coordinates are millimetres. All connectivity is explicit wires and global labels;
the exported KiCad netlist is the authority for checking the resulting circuit.
"""
from pathlib import Path
import copy
import json
import uuid
import sexpdata as sx

S = sx.Symbol
REVISION = 'A0.3-WIP'

def n(tag, *args):
    return [S(tag), *args]

def uid(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'little-guy-v2/pcb/' + name))

def children(node, tag):
    return [x for x in node if isinstance(x, list) and x and x[0] == S(tag)]

def pretty(node, depth=0):
    if not isinstance(node, list) or not any(isinstance(x, list) for x in node):
        return sx.dumps(node)
    out = '(' + ' '.join(sx.dumps(x) for x in node if not isinstance(x, list))
    for x in node:
        if isinstance(x, list):
            out += '\n' + '  ' * (depth + 1) + pretty(x, depth + 1)
    return out + '\n' + '  ' * depth + ')'

def effects(size=1.0, justify=None):
    e = n('effects', n('font', n('size', size, size)))
    if justify:
        e.append(n('justify', S(justify)))
    return e

def prop(name, value, x=0, y=0, hidden=False):
    p = n('property', name, str(value), n('at', x, y, 0), effects())
    if hidden:
        p.append(n('hide', S('yes')))
    return p

def ic(name, left, right, width=12.7):
    """Pin tuple is number, name, electrical type. No implicit pins."""
    h = (max(len(left), len(right)) + 1) * 1.27
    sym = n('symbol', name, n('pin_names', n('offset', 0.762)),
            n('in_bom', S('yes')), n('on_board', S('yes')),
            prop('Reference', 'U', 0, h + 5.08), prop('Value', name, 0, h + 2.54),
            prop('Footprint', '', hidden=True), prop('Datasheet', '', hidden=True))
    sym.append(n('symbol', name + '_0_1',
                 n('rectangle', n('start', -width, h), n('end', width, -h),
                   n('stroke', n('width', .254), n('type', S('default'))),
                   n('fill', n('type', S('background'))))))
    pins = []
    for side, group in enumerate([left, right]):
        for j, (number, label, kind) in enumerate(group):
            y = (len(group) - 1) * 1.27 - 2.54 * j
            x = (-1 if side == 0 else 1) * (width + 5.08)
            pins.append(n('pin', S(kind), S('line'), n('at', x, y, 0 if side == 0 else 180),
                          n('length', 5.08), n('name', label, effects(.9)),
                          n('number', str(number), effects(.9))))
    sym.append(n('symbol', name + '_1_1', *pins))
    return sym

def two_pin(kind):
    # Library geometry is local; y is inverted when symbols are instantiated.
    body = n('rectangle', n('start', -1.016, 2.54), n('end', 1.016, -2.54),
             n('stroke', n('width', .254), n('type', S('default'))),
             n('fill', n('type', S('none'))))
    if kind == 'C':
        body = n('polyline', n('pts', n('xy', -2.032, .762), n('xy', 2.032, .762)),
                 n('stroke', n('width', .254), n('type', S('default'))),
                 n('fill', n('type', S('none'))))
    a = n('symbol', kind, n('pin_names', n('offset', 0), S('hide')),
          n('pin_numbers', S('hide')), n('in_bom', S('yes')), n('on_board', S('yes')),
          prop('Reference', kind, 3.81, 1.27), prop('Value', kind, 3.81, -1.27),
          prop('Footprint', '', hidden=True), prop('Datasheet', '', hidden=True))
    graphics = [body]
    if kind == 'C':
        graphics.append(n('polyline', n('pts', n('xy', -2.032, -.762), n('xy', 2.032, -.762)),
                          n('stroke', n('width', .254), n('type', S('default'))),
                          n('fill', n('type', S('none')))))
    a.append(n('symbol', kind + '_0_1', *graphics))
    a.append(n('symbol', kind + '_1_1', *[
        n('pin', S('passive'), S('line'), n('at', 0, y, angle), n('length', length),
          n('name', '~', effects()), n('number', number, effects()))
        for number, y, angle, length in [('1', 5.08, 270, 4.318 if kind == 'C' else 2.54),
                                          ('2', -5.08, 90, 4.318 if kind == 'C' else 2.54)]]))
    return a

class Sheet:
    def __init__(self, name, title, page, project='little-guy-power'):
        self.name, self.project, self.page = name, project, page
        self.rootid = uid(project + '/root')
        self.sheetid = uid(project + '/' + name)
        self.defs, self.items, self.parts = {}, [], []
        self.title = title

    def text(self, text, x, y, size=1.2):
        self.items.append(n('text', text, n('at', x, y, 0), effects(size, 'left'),
                            n('uuid', uid(self.name + '/text/' + text))))

    def wire(self, x1, y1, x2, y2):
        self.items.append(n('wire', n('pts', n('xy', x1, y1), n('xy', x2, y2)),
                            n('stroke', n('width', 0), n('type', S('default'))),
                            n('uuid', uid(f'{self.name}/wire/{x1}/{y1}/{x2}/{y2}'))))

    def label(self, net, x, y, angle=0):
        self.items.append(n('global_label', net, n('shape', S('bidirectional')),
                            n('at', x, y, angle), effects(.9, 'left'),
                            n('uuid', uid(f'{self.name}/label/{net}/{x}/{y}'))))

    def part(self, definition, ref, value, x, y, nets, footprint='', mpn='', datasheet='', dnp=False):
        x, y = round(round(x / 1.27) * 1.27, 4), round(round(y / 1.27) * 1.27, 4)
        symbol = copy.deepcopy(definition)
        name = symbol[1]
        symbol[1] = 'LGV2:' + name
        self.defs[name] = symbol
        ys = []
        pins = []
        for sub in children(definition, 'symbol'):
            for pin in children(sub, 'pin'):
                num = str(children(pin, 'number')[0][1])
                at = children(pin, 'at')[0]
                px, py, angle = at[1:4]
                ys.append(py)
                pins.append((num, px, py, angle))
        h = max(ys, default=5.08) + 5.08
        passive = name in ['R', 'C', 'L']
        rx, ry = (x + 6.35, y - 1.27) if passive else (x, y - h - 2.54)
        vx, vy = (x + 6.35, y + 1.27) if passive else (x, y - h)
        self.items.append(n('symbol', n('lib_id', 'LGV2:' + name), n('at', x, y, 0),
                            n('unit', 1), n('in_bom', S('yes')), n('on_board', S('yes')),
                            n('dnp', S('yes' if dnp else 'no')), n('uuid', uid(self.name + '/' + ref)),
                            prop('Reference', ref, rx, ry), prop('Value', value, vx, vy),
                            prop('Footprint', footprint, x, y, True), prop('Datasheet', datasheet, x, y, True),
                            prop('MPN', mpn, x, y, True),
                            prop('Selection', 'candidate' if mpn else 'MPN_PENDING', x, y, True),
                            n('instances', n('project', self.project,
                              n('path', '/' + self.rootid + '/' + self.sheetid,
                                n('reference', ref), n('unit', 1))))))
        assert set(nets) == {p[0] for p in pins}, (ref, set(nets) ^ {p[0] for p in pins})
        for number, px, py, angle in pins:
            ax, ay = round(x + px, 4), round(y - py, 4)
            net = nets[number]
            if net is None:
                self.items.append(n('no_connect', n('at', ax, ay), n('uuid', uid(self.name + '/' + ref + '/nc/' + number))))
            else:
                dx = -7.62 if angle == 0 else 7.62 if angle == 180 else 0
                dy = -3.81 if angle == 270 else 3.81 if angle == 90 else 0
                self.wire(ax, ay, round(ax + dx, 4), round(ay + dy, 4))
                self.label(net, round(ax + dx, 4), round(ay + dy, 4), 0 if dx < 0 else 180 if dx > 0 else 90)
        self.parts.append(dict(ref=ref, value=value, footprint=footprint, mpn=mpn,
                               datasheet=datasheet, pins=nets, sheet=self.name, dnp=dnp))

    def passive(self, kind, ref, value, net1, net2, x, y, footprint='', mpn=''):
        self.part(two_pin(kind), ref, value, x, y, {'1': net1, '2': net2}, footprint, mpn)

    def connector(self, ref, name, nets, x, y):
        defs = ic('Header_' + str(len(nets)), [(str(i + 1), str(i + 1), 'passive') for i in range(len(nets))], [], 3.81)
        self.part(defs, ref, name, x, y, {str(i + 1): net for i, net in enumerate(nets)})

    def power_flag(self, net, x, y):
        flag = ic('External_Power_Source', [(1, 'SOURCE', 'power_out')], [], 2.54)
        self.part(flag, '#FLG' + str(self.page * 100 + len(self.parts)), net, x, y, {'1': net})
        # Source declarations are schematic annotations, not BOM/PCB components.
        item = self.items[-3]
        if isinstance(item, list) and item[0] == S('symbol'):
            for tag in ['in_bom', 'on_board']:
                children(item, tag)[0][1] = S('no')
        self.parts[-1]['annotation'] = True

    def save(self, directory):
        root = n('kicad_sch', n('version', 20260306), n('generator', S('eeschema')),
                 n('generator_version', '10.0'), n('uuid', self.sheetid), n('paper', 'A3'),
                 n('title_block', n('title', self.title), n('date', '2026-10-25'), n('rev', REVISION),
                   n('comment', 1, 'DESIGN IN PROGRESS - NOT FOR FABRICATION')),
                 n('lib_symbols', *self.defs.values()), *self.items,
                 n('embedded_fonts', S('no')))
        (directory / (self.name + '.kicad_sch')).write_text(pretty(root) + '\n')

def save_project(directory, sheets, title=None, status=None):
    directory.mkdir(parents=True, exist_ok=True)
    project = sheets[0].project
    rootid = uid(project + '/root')
    root = n('kicad_sch', n('version', 20260306), n('generator', S('eeschema')),
             n('generator_version', '10.0'), n('uuid', rootid), n('paper', 'A3'),
             n('title_block', n('title', title or 'Little Guy V2 - power board'), n('date', '2026-10-25'),
               n('rev', REVISION), n('comment', 1, 'INCOMPLETE - NOT FOR FABRICATION')),
             n('lib_symbols'))
    root.append(n('text', status or 'POWER BOARD / CIRCUIT CAPTURE IN PROGRESS\nPack protection, transient handling, exact parts and physical tests remain open.\nDo not order, connect a battery pack, or treat ERC as physical validation.',
                  n('at', 20, 20, 0), effects(1.4, 'left'), n('uuid', uid(project + '/warning'))))
    for i, sheet in enumerate(sheets):
        x, y = 15 + (i % 4) * 100, 55 + (i // 4) * 55
        sheetname = prop('Sheetname', sheet.title, x + 2, y - 1.27)
        sheetfile = prop('Sheetfile', sheet.name + '.kicad_sch', x + 2, y + 32.54)
        for field in [sheetname,sheetfile]:
            field[-1] = effects(.9, 'left')
        root.append(n('sheet', n('at', x, y), n('size', 90, 30),
                      n('stroke', n('width', .254), n('type', S('default'))),
                      n('fill', n('color', 0, 0, 0, 0)), n('uuid', sheet.sheetid),
                      sheetname, sheetfile,
                      n('instances', n('project', project, n('path', '/' + rootid, n('page', str(i + 2)))))))
        sheet.save(directory)
    root.append(n('sheet_instances', n('path', '/', n('page', '1'))))
    root.append(n('embedded_fonts', S('no')))
    (directory / (project + '.kicad_sch')).write_text(pretty(root) + '\n')
    (directory / (project + '.kicad_pro')).write_text(json.dumps({'meta': {'filename': project + '.kicad_pro', 'version': 1}}, indent=2) + '\n')
    definitions = {}
    for sheet in sheets:
        for key, definition in sheet.defs.items():
            definition = copy.deepcopy(definition)
            definition[1] = key
            definitions[key] = definition
    library = n('kicad_symbol_lib', n('version', 20241209), n('generator', S('kicad_symbol_editor')), *definitions.values())
    (directory / 'LGV2.kicad_sym').write_text(pretty(library) + '\n')
    (directory / 'sym-lib-table').write_text('(sym_lib_table (lib (name "LGV2") (type "KiCad") (uri "${KIPRJMOD}/LGV2.kicad_sym") (options "") (descr "V2 circuit symbols; pin maps from linked manufacturer documentation")))\n')
    fplibs = sorted({p['footprint'].split(':')[0] for s in sheets for p in s.parts if p['footprint']})
    table = n('fp_lib_table', *[n('lib', n('name', lib), n('type', 'KiCad'),
              n('uri', ('${KIPRJMOD}/' if lib=='LGV2_Footprints' else '${KICAD10_FOOTPRINT_DIR}/') + lib + '.pretty'),
              n('options', ''), n('descr', 'Local candidate footprints' if lib=='LGV2_Footprints' else 'KiCad standard footprint library')) for lib in fplibs])
    (directory / 'fp-lib-table').write_text(pretty(table) + '\n')
    parts = [p for s in sheets for p in s.parts]
    (directory / 'circuit.json').write_text(json.dumps({'revision': REVISION, 'release': False, 'parts': parts}, indent=2) + '\n')
    print(f'{project}: {len(sheets)} circuit sheets, {len(parts)} components')

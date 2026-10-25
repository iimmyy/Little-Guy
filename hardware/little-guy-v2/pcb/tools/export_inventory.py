"""Export unreleased inventory and electrical port tables from the circuit manifests."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]
parts=[];ports=[]
for board in ['power','spinal','foot','head']:
    count=2 if board=='foot' else 1
    manifest=json.loads((ROOT/board/'circuit.json').read_text())
    for p in manifest['parts']:
        if p.get('annotation'):continue
        parts.append({'board':board,'ref':p['ref'],'value_and_notes':p['value'],'quantity_per_board':1,
                      'boards_in_robot':count,'mpn':p['mpn'],'footprint':p['footprint'],
                      'selection_status':'candidate' if p['mpn'] else 'MPN_PENDING',
                      'footprint_status':'candidate' if p['footprint'] else 'FOOTPRINT_PENDING',
                      'released':False,'dnp':p['dnp'],'datasheet':p['datasheet'],'sheet':p['sheet']})
        if p['ref'].startswith('J'):
            for pin,net in p['pins'].items():
                ports.append({'board':board,'ref':p['ref'],'function':p['value'],'pin':pin,'net':net,
                              'connector_mpn':'PENDING','cable_released':False})
for name,rows in [('bom-draft.csv',parts),('interfaces.csv',ports)]:
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(str(len(parts))+' physical schematic entries exported; all unreleased.')

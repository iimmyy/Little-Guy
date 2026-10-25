"""Inspect pinned, unmodified SO-101 mechanical references; no fit certification.

Uses the same OCP environment as inspect_servo_step.py. The optional mesh is only
for a reference visualization. Names and placements come from the STEP assembly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_SOLID
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from inspect_servo_step import bbox, children, circles, mesh

ROOT = Path(__file__).resolve().parents[1]/"reference/so101"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mesh-json",type=Path)
    args=parser.parse_args()
    rows=[]; meshes=[]
    for path in sorted((ROOT/"STEP/SO101").glob("*.step")):
        if path.name == "SO101 Assembly.step":
            continue
        reader=STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise RuntimeError(f"Cannot read {path}")
        reader.TransferRoots();shape=reader.OneShape()
        rows.append({"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bounds":bbox(shape),"valid":BRepCheck_Analyzer(shape).IsValid(),
                     "solids":len(list(children(shape,TopAbs_SOLID))),"circles":circles(shape)})
        if args.mesh_json:
            meshes.append({"name":path.stem,"triangles":mesh(shape)})
    (ROOT/"inspection.json").write_text(json.dumps(rows,indent=2)+"\n")
    if args.mesh_json:
        args.mesh_json.parent.mkdir(parents=True,exist_ok=True)
        args.mesh_json.write_text(json.dumps(meshes))

    path=ROOT/"STEP/SO101/SO101 Assembly.step"
    reader=STEPCAFControl_Reader();reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read {path}")
    document=TDocStd_Document(TCollection_ExtendedString("reference"))
    if not reader.Transfer(document):
        raise RuntimeError("Cannot transfer assembly")
    tool=XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots=TDF_LabelSequence();tool.GetFreeShapes(roots)
    if roots.Length()!=1:
        raise RuntimeError("Expected one assembly root")
    seq=TDF_LabelSequence();tool.GetComponents_s(roots.Value(1),seq)
    selected=[]
    for i in range(1,seq.Length()+1):
        label=seq.Value(i);attribute=TDataStd_Name()
        if not label.FindAttribute(TDataStd_Name.GetID_s(),attribute):
            continue
        name=attribute.Get().ToExtString()
        if not name.startswith(("ST3215 Servo", "Rotation_Pitch", "Upper_arm", "Motor_holder_SO101_Base")):
            continue
        shape=tool.GetShape_s(label)
        horn_circles=[c for c in circles(shape) if abs(c["radius_mm"]-9.6)<1e-4]
        centres=[c["center_xyz_mm"] for c in horn_circles]
        span=max((math.dist(a,b) for a in centres for b in centres),default=0)
        selected.append({"name":name,"bounds":bbox(shape),"horn_outer_circles":horn_circles,
                         "maximum_circle_centre_separation_mm":span,
                         "span_note":"For four coaxial outer circles, maximum separation measures the two fittings' outer-face spacing. With one fitting only this is its thickness."})
    report={"assembly_source":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "method":"XCAF named top-level component shapes retain their assembly placements; radius-9.6 circular edge centres are reported without modifying geometry.",
            "scope":"Selected source geometry only; no whole-assembly validity, mating clearance, printed strength or purchased-hardware fit asserted.",
            "components":selected,"physical_fit_verified":False}
    (ROOT/"assembly-inspection.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"individual_validity":{r["file"]:r["valid"] for r in rows},
                      "servo_fitting_spans_mm":{r["name"]:r["maximum_circle_centre_separation_mm"] for r in selected if r["name"].startswith("ST3215")}},indent=2))


if __name__=="__main__":
    main()

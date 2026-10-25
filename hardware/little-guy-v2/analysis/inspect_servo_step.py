"""Inspect the unmodified vendor STEP; never heal or overwrite its geometry.

Optional --mesh-json creates a disposable visualization mesh, not manufacturing
geometry. All numerical dimensions here describe this file, not physical parts.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Circle
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference/ST3215-3D/ST3215.step"


def bbox(shape):
    box = Bnd_Box()
    # Ignore display triangulations so optional rendering cannot change bounds.
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    v = box.Get()
    return {"min_xyz_mm": list(v[:3]), "max_xyz_mm": list(v[3:]),
            "size_xyz_mm": [v[i+3]-v[i] for i in range(3)]}


def children(shape, kind):
    explorer=TopExp_Explorer(shape, kind)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def circles(shape):
    result=set()
    for edge in children(shape, TopAbs_EDGE):
        curve=BRepAdaptor_Curve(TopoDS.Edge_s(edge))
        if curve.GetType()!=GeomAbs_Circle:
            continue
        circle=curve.Circle(); point=circle.Location(); axis=circle.Axis().Direction()
        result.add(tuple(round(x,6) for x in
                         (circle.Radius(),point.X(),point.Y(),point.Z(),axis.X(),axis.Y(),axis.Z())))
    return [{"radius_mm":row[0],"center_xyz_mm":list(row[1:4]),"axis_direction":list(row[4:])}
            for row in sorted(result)]


def mesh(shape):
    # Chosen display tolerances only; not tolerances of the physical component.
    mesher=BRepMesh_IncrementalMesh(shape,0.12,False,0.35,False)
    mesher.Perform()
    triangles=[]
    for face in children(shape,TopAbs_FACE):
        face=TopoDS.Face_s(face); location=TopLoc_Location()
        tri=BRep_Tool.Triangulation_s(face,location)
        if tri is None:
            continue
        for i in range(1,tri.NbTriangles()+1):
            indices=list(tri.Triangle(i).Get())
            if face.Orientation()==TopAbs_REVERSED:
                indices.reverse()
            points=[]
            for j in indices:
                p=tri.Node(j).Transformed(location.Transformation())
                points.append([p.X(),p.Y(),p.Z()])
            triangles.append(points)
    return triangles


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mesh-json",type=Path)
    args=parser.parse_args()
    data=SOURCE.read_bytes()
    reader=STEPControl_Reader()
    status=reader.ReadFile(str(SOURCE))
    if status!=IFSelect_RetDone:
        raise RuntimeError(f"STEP reader returned {status}")
    roots=reader.TransferRoots(); shape=reader.OneShape()
    analyzer=BRepCheck_Analyzer(shape)
    solids=list(children(shape,TopAbs_SOLID))
    rows=[]; meshes=[]
    for i,solid in enumerate(solids):
        valid=analyzer.IsValid(solid)
        row={"index":i,**bbox(solid),"brep_valid":valid,
             "brep_status":[str(s) for s in analyzer.Result(solid).Status()]}
        # These are the two round fittings in this source's traversal order.
        # Identification is supported by their 19.2 mm circular outer envelopes.
        if i in (6,7):
            row["circular_edges"]=circles(solid)
        rows.append(row)
        if args.mesh_json:
            meshes.append({"solid_index":i,"triangles":mesh(solid)})
    report={"source":str(SOURCE.relative_to(ROOT)),"sha256":hashlib.sha256(data).hexdigest(),
            "ocp_package_version":importlib.metadata.version("cadquery-ocp-novtk"),
            "units":"mm: STEP geometric context uses SI_UNIT(.MILLI.,.METRE.)",
            "product_names":re.findall(r"PRODUCT\('([^']*)'",data.decode("utf-8")),
            "transferred_roots":roots,"solid_count":len(solids),
            "assembly_brep_valid":analyzer.IsValid(),"assembly_bounds":bbox(shape),"solids":rows,
            "geometry_modified":False,"physical_part_verified":False,
            "scope":"Numerical inspection of the supplied SCS215-titled STEP only; not proof of ST-3215-C018 fit"}
    (ROOT/"reference/servo-step-inspection.json").write_text(json.dumps(report,indent=2)+"\n")
    if args.mesh_json:
        args.mesh_json.parent.mkdir(parents=True,exist_ok=True)
        args.mesh_json.write_text(json.dumps(meshes))
    print(json.dumps({"solid_count":len(solids),"valid":report["assembly_brep_valid"],
                      "size_xyz_mm":report["assembly_bounds"]["size_xyz_mm"],
                      "invalid_solid_indices":[r["index"] for r in rows if not r["brep_valid"]]},indent=2))


if __name__=="__main__":
    main()

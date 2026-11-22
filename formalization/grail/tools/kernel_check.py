"""Standalone sequential Lean compilation and transitive axiom checks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-root", type=Path, default=ROOT/".lake/packages")
    args = parser.parse_args()
    dependencies = args.dependency_root.resolve()
    manifest = json.loads((ROOT/"lake-manifest.json").read_text())
    mathlib = next(p for p in manifest["packages"] if p["name"] == "mathlib")
    actual = subprocess.check_output(["git", "-C", str(dependencies/"mathlib"),
                                      "rev-parse", "HEAD"], text=True).strip()
    if actual != mathlib["rev"]:
        raise SystemExit("Mathlib revision differs from the pinned manifest")
    compiler = shutil.which("lean")
    version = subprocess.check_output([compiler, "--version"], cwd=ROOT, text=True)
    if "4.36.0-rc2" not in version:
        raise SystemExit("Use the Lean version in lean-toolchain")
    build = ROOT/".lake/build/lib/lean"
    build.mkdir(parents=True, exist_ok=True)
    libraries = [build]+[p/".lake/build/lib/lean" for p in sorted(dependencies.iterdir())
                         if (p/".lake/build/lib/lean").is_dir()]
    environment = dict(os.environ, LEAN_PATH=os.pathsep.join(map(str, libraries)))
    registry = json.loads((ROOT/"theorems.json").read_text())
    logs = ROOT/".lake/verification"
    logs.mkdir(parents=True, exist_ok=True)
    sources = {}
    for module in registry["build_order"]:
        source = ROOT/"lean"/(module.replace(".", "/")+".lean")
        target = build/(module.replace(".", "/")+".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        with (logs/(module+".log")).open("w") as log:
            subprocess.run([compiler, "-R", "lean", "-j1", "-M8192", "-DmaxHeartbeats=800000", "-o", str(target), str(source)],
                           cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        print("Compiled", module, flush=True)
    names = list(dict.fromkeys(name for t in registry["theorems"]
                              for name in t["lean_declarations"]+t["supporting_declarations"]))
    check = logs/"CheckAxioms.lean"
    check.write_text("import Grail\n"+"".join("#print axioms "+name+"\n" for name in names))
    output = subprocess.check_output([compiler, "-j1", "-M8192", str(check)],
                                      cwd=ROOT, env=environment, text=True)
    (logs/"axioms.log").write_text(output)
    footprints = re.findall(r"depends on axioms:\s*\[([^]]*)\]", output)
    if len(footprints) != len(names):
        raise SystemExit("Missing declaration axiom footprints")
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    for footprint in footprints:
        if not {name.strip() for name in footprint.split(",") if name.strip()} <= allowed:
            raise SystemExit("Unexpected axiom footprint: "+footprint)
    report = {"passed": True, "lean_version": version.strip(), "mathlib_revision": actual,
              "sources": sources, "headline_declarations_checked": names,
              "allowed_axioms": sorted(allowed), "modules_compiled": len(sources)}
    (logs/"kernel_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print("PASS: standalone Lean kernel and transitive axiom checks.")


if __name__ == "__main__":
    main()

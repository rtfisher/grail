"""Build only the public user manual and technical reference guide."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    for folder, name in [("user_manual", "grail_user_manual"),
                         ("reference_guide", "grail_reference_guide")]:
        directory = ROOT/"docs"/folder
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        name+".tex"], cwd=directory, check=True)
        log = (directory/(name+".log")).read_text(errors="replace")
        if re.search(r"There were undefined references|Citation .+ undefined|Reference .+ undefined", log):
            raise SystemExit("Unresolved guide reference: "+name)
        print("Built", directory/(name+".pdf"), flush=True)


if __name__ == "__main__":
    main()

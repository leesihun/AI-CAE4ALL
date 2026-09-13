"""Print the paper's section outline from source."""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = ["introduction.tex", "benchmarks.tex", "method.tex",
         "protocol.tex", "experiments.tex", "conclusion.tex"]

PAT = re.compile(r"\\(sub)*section\*?\{(.+?)\}")

n_sec = n_sub = n_ssub = 0
for f in FILES:
    try:
        s = io.open(f, encoding="utf-8").read()
    except OSError:
        continue
    s = re.sub(r"(?<!\\)%.*", "", s)          # drop comments
    for m in PAT.finditer(s):
        lvl = (m.group(0).count("sub"))
        title = m.group(2)
        if lvl == 0:
            n_sec += 1
            print("%d  %s" % (n_sec, title))
            n_sub = 0
        elif lvl == 1:
            n_sub += 1
            n_ssub = 0
            print("   %d.%d  %s" % (n_sec, n_sub, title))
        else:
            n_ssub += 1
            print("        %d.%d.%d  %s" % (n_sec, n_sub, n_ssub, title))

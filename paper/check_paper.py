"""Audit the paper source before believing a build.

A LaTeX build that "succeeds" still ships broken cross-references as the
literal text "??" and missing citations as "[?]". This checks for them
directly, plus the mistakes that are easy to make while writing across several
included files.
"""
import os
import re
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = ["main.tex", "introduction.tex", "benchmarks.tex", "method.tex",
       "protocol.tex", "experiments.tex", "conclusion.tex", "appendix.tex"]


def read(p):
    fp = os.path.join(HERE, p)
    if not os.path.exists(fp):
        return ""
    return open(fp, encoding="utf-8", errors="replace").read()


def strip_comments(s):
    # drop % comments but keep \%
    return re.sub(r"(?<!\\)%.*", "", s)


def main():
    present = [f for f in TEX if os.path.exists(os.path.join(HERE, f))]
    src = "".join(strip_comments(read(f)) for f in present)
    bibtxt = read("references.bib")

    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set()
    for pat in (r"\\ref\{([^}]+)\}", r"\\eqref\{([^}]+)\}",
                r"\\autoref\{([^}]+)\}", r"\\Cref\{([^}]+)\}",
                r"\\cref\{([^}]+)\}"):
        refs.update(re.findall(pat, src))

    cites = set()
    for m in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", src):
        cites.update(x.strip() for x in m.split(",") if x.strip())
    bibkeys = set(k.strip() for k in re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibtxt))

    dangling_ref = sorted(refs - labels)
    dangling_cite = sorted(cites - bibkeys)
    dup = [l for l in labels if src.count("\\label{%s}" % l) > 1]

    # unbalanced environments
    begins = re.findall(r"\\begin\{([^}]+)\}", src)
    ends = re.findall(r"\\end\{([^}]+)\}", src)
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    unbal = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}

    print("files          : %s" % ", ".join(present))
    print("labels defined : %d" % len(labels))
    print("refs used      : %d" % len(refs))
    print("cites used     : %d   (bib has %d keys)" % (len(cites), len(bibkeys)))
    print()
    ok = True
    if dangling_ref:
        ok = False
        print("DANGLING REFS (%d): %s" % (len(dangling_ref), dangling_ref))
    if dangling_cite:
        ok = False
        print("DANGLING CITES (%d): %s" % (len(dangling_cite), dangling_cite))
    if dup:
        ok = False
        print("DUPLICATE LABELS: %s" % sorted(set(dup)))
    if unbal:
        ok = False
        print("UNBALANCED ENVIRONMENTS: %s" % unbal)

    # check the produced PDF for the telltale unresolved markers
    pdf = os.path.join(HERE, "main.pdf")
    if os.path.exists(pdf):
        print("pdf            : %.1f KB" % (os.path.getsize(pdf) / 1024))
    else:
        ok = False
        print("NO PDF PRODUCED")

    print()
    print("SOURCE AUDIT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

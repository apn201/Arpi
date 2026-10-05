"""Flag characters that are hard to type on a keyboard in user-facing text:
dashes other than '-', curly quotes, the ellipsis character, arrows, bullets,
checkmarks, non-breaking spaces. Also catches them written as \\u escapes in
JavaScript strings, which is how they slip in unnoticed.

    python tools/check_chars.py                  # page, report and agent
    python tools/check_chars.py README.md docs/report.md

Exit 1 if anything is found. Code comments are checked too: they end up in a
public repo that judges read.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["web/index.html", "src/arpi/report.py", "src/arpi/agent.py",
           "src/arpi/live.py", "src/arpi/cascade.py"]

BAD = {0x2012, 0x2013, 0x2014, 0x2015, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026,
       0x2022, 0x00A0, 0x200B, 0x00AD, 0x00D7, 0x2248, 0x2713, 0x2714, 0x25B8,
       0x25B6, 0x2192, 0x2190, 0x21D2}
ESCAPE = re.compile(r"\\u(201[2-9a-dA-D]|2026|2022|00a0|2713|25b8|2192)", re.I)


def main(paths):
    hits = 0
    for rel in paths or DEFAULT:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            found = [c for c in line if ord(c) in BAD] + ESCAPE.findall(line)
            for f in found:
                hits += 1
                print("{}:{}: {!r}  {}".format(rel, n, f, line.strip()[:90]))
    print("{} found".format(hits))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

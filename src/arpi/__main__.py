import argparse
import json
import sys

from .decode import as_dict, load_known, scan


def main():
    ap = argparse.ArgumentParser(prog="arpi",
                                 description="Read a damaged EAN-13 / UPC-A.")
    ap.add_argument("images", nargs="+", help="one or more frames of one label")
    ap.add_argument("--known", help="CSV or newline list of valid codes")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-text", action="store_true",
                    help="bars only, ignore the printed digits")
    args = ap.parse_args()

    known = load_known(args.known) if args.known else None
    session = scan(args.images, known_codes=known, use_text=not args.no_text)
    result = session.result()
    cmap = session.latest.reading.cmap if session.latest else None
    out = as_dict(result, cmap, session)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(out["message"])
    for c in out["candidates"]:
        print("  {}  {:>6.1%}  {}  ({})".format(c["code"], c["share"],
                                               c["symbology"], c["prefix"]))
    return 0 if out["candidates"] else 1


if __name__ == "__main__":
    sys.exit(main())

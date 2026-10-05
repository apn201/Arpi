"""EAN-13 and UPC-A, from the specification and nothing else.

95 modules, left to right:

    0-2     start guard   101
    3-44    six left digits, 7 modules each, L or G encoded
    45-49   centre guard  01010
    50-91   six right digits, 7 modules each, R encoded
    92-94   end guard     101

The 13th digit (the leading one) is never drawn as bars. It is carried by the
L/G parity pattern across the six left digits. UPC-A is EAN-13 with a leading
zero, which is parity LLLLLL.
"""

MODULES = 95

L_CODES = ["0001101", "0011001", "0010011", "0111101", "0100011",
           "0110001", "0101111", "0111011", "0110111", "0001011"]
R_CODES = ["".join("1" if b == "0" else "0" for b in c) for c in L_CODES]
G_CODES = [c[::-1] for c in R_CODES]

# Leading digit -> parity of left digits 1..6. 0 = L (odd), 1 = G (even).
PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
          "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]
PARITY_BITS = [int(p.replace("L", "0").replace("G", "1"), 2) for p in PARITY]
# 64 possible bit patterns, 10 legal. -1 means the left half cannot be EAN-13.
LEADING_FROM_BITS = [-1] * 64
for _d, _bits in enumerate(PARITY_BITS):
    LEADING_FROM_BITS[_bits] = _d

START = 0
LEFT = 3
CENTRE = 45
RIGHT = 50
END = 92

GUARDS = {0: 1, 1: 0, 2: 1,
          45: 0, 46: 1, 47: 0, 48: 1, 49: 0,
          92: 1, 93: 0, 94: 1}


def digit_start(position):
    """Module index of encoded digit `position` (0..11). Position 0 is the
    second digit of the printed number, because the first is not drawn."""
    return LEFT + 7 * position if position < 6 else RIGHT + 7 * (position - 6)


def known_modules():
    """Every module whose colour the specification fixes regardless of the
    number. Guards, plus the first and last module of every digit: L and G
    codes all start with a space and end with a bar, R codes the reverse.

    35 of 95. This is what the scanner fits its geometry against, so a code
    with the guards destroyed can still be located from its digit edges.
    """
    known = dict(GUARDS)
    for pos in range(12):
        s = digit_start(pos)
        left = pos < 6
        known[s] = 0 if left else 1
        known[s + 6] = 1 if left else 0
    return known


def weight(index):
    """Checksum weight of digit `index` in the 13-digit number."""
    return 1 if index % 2 == 0 else 3


def check_digit(first12):
    total = sum(weight(i) * int(c) for i, c in enumerate(first12))
    return (10 - total % 10) % 10


def is_valid(code):
    return (len(code) == 13 and code.isdigit()
            and check_digit(code[:12]) == int(code[12]))


def normalise(code):
    """Accept a 12-digit UPC-A or a 13-digit EAN-13, return 13 digits."""
    code = "".join(c for c in str(code) if c.isdigit())
    if len(code) == 12:
        code = "0" + code
    if len(code) != 13:
        raise ValueError("expected 12 or 13 digits, got {!r}".format(code))
    return code


def complete(first12):
    return first12 + str(check_digit(first12))


def encode(code):
    """13 digits -> 95-character module string, '1' is a bar."""
    code = normalise(code)
    if not is_valid(code):
        raise ValueError("bad check digit: {}".format(code))
    parity = PARITY[int(code[0])]
    out = ["101"]
    for i, c in enumerate(code[1:7]):
        out.append((L_CODES if parity[i] == "L" else G_CODES)[int(c)])
    out.append("01010")
    for c in code[7:]:
        out.append(R_CODES[int(c)])
    out.append("101")
    bits = "".join(out)
    assert len(bits) == MODULES
    return bits


def symbology(code):
    return "UPC-A" if code[0] == "0" else "EAN-13"

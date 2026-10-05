"""Synthetic damaged barcodes with exact ground truth.

For scale and for sweeping damage against recovery. It does not replace the
physical set and the report must never present it as if it did: rendered
damage is cleaner than real damage, and every number from here is labelled
synthetic.

Two stages, in the order the world applies them:

    label damage    happens to the paper: scratch, tear, occlusion, smear, fade
    camera          happens to the photo: perspective, glare, blur, noise, JPEG

Every image comes back with the fraction of modules the label damage touched,
measured from the damage mask rather than from the severity knob, so the sweep
plots what actually happened to the symbol.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ean13

SS = 4                 # supersampling, so non-integer module widths antialias
QUIET = 11             # quiet zone in modules, both sides (spec minimum 11/7)


@dataclass
class Label:
    image: np.ndarray          # uint8 gray
    code: str
    module_px: float
    x0: float                  # left edge of module 0, in pixels
    bar_top: int
    bar_bottom: int
    damage_mask: np.ndarray = None
    damage: list = field(default_factory=list)
    # 1 where there is paper, 0 where a piece is missing and whatever the
    # label is lying on shows through. None means intact.
    paper: np.ndarray = None

    def damaged_fraction(self, threshold=0.3):
        """Share of the 95 modules whose bar area the damage covered by more
        than `threshold`."""
        if self.damage_mask is None:
            return 0.0
        band = self.damage_mask[self.bar_top:self.bar_bottom].astype(float)
        hit = 0
        for m in range(ean13.MODULES):
            a = int(round(self.x0 + m * self.module_px))
            b = max(a + 1, int(round(self.x0 + (m + 1) * self.module_px)))
            if band[:, a:b].mean() > threshold:
                hit += 1
        return hit / ean13.MODULES


def render(code, module_px=3.0, height_modules=50, text=True):
    code = ean13.normalise(code)
    bits = ean13.encode(code)
    mp = module_px * SS
    w = int(round((ean13.MODULES + 2 * QUIET) * mp))
    bar_h = int(round(height_modules * mp))
    text_h = int(round(9 * mp)) if text else int(round(2 * mp))
    margin = int(round(3 * mp))
    h = margin + bar_h + text_h + margin
    img = np.full((h, w), 255, np.uint8)
    x0 = QUIET * mp
    guards = set(ean13.GUARDS)
    for m, b in enumerate(bits):
        if b != "1":
            continue
        a = int(round(x0 + m * mp))
        e = int(round(x0 + (m + 1) * mp))
        bottom = margin + bar_h + (int(round(5 * mp)) if m in guards else 0)
        img[margin:bottom, a:e] = 0
    if text:
        scale = 6 * mp / 22.0          # Hershey simplex is ~22 px tall at 1.0
        y = margin + bar_h + int(round(7.5 * mp))
        th = max(1, int(mp * 0.9))
        cv2.putText(img, code[0], (int(x0 - 8 * mp), y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)
        for i, c in enumerate(code[1:]):
            start = ean13.digit_start(i)
            cv2.putText(img, c, (int(x0 + (start + 1.2) * mp), y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)
    small = cv2.resize(img, (w // SS, h // SS), interpolation=cv2.INTER_AREA)
    return Label(image=small, code=code, module_px=module_px,
                 x0=QUIET * module_px, bar_top=margin // SS,
                 bar_bottom=(margin + bar_h) // SS,
                 damage_mask=np.zeros(small.shape, bool))


# ---- label damage ------------------------------------------------------------

def _bar_span(label):
    a = label.x0
    b = label.x0 + ean13.MODULES * label.module_px
    return a, b


def scratch(label, rng, severity=0.3):
    """Thin lines through the ink, mostly white. Rarely crosses every row,
    which is exactly what multi-row voting is for."""
    img, mask = label.image, label.damage_mask
    a, b = _bar_span(label)
    n = 1 + int(severity * 8)
    for _ in range(n):
        x1 = rng.uniform(a, b)
        y1 = rng.uniform(label.bar_top, label.bar_bottom)
        ang = rng.uniform(-np.pi / 3, np.pi / 3)
        length = rng.uniform(0.3, 1.2) * (label.bar_bottom - label.bar_top)
        x2 = x1 + length * np.sin(ang) * rng.choice([-1, 1]) * 2
        y2 = y1 + length * np.cos(ang)
        t = max(1, int(round(rng.uniform(0.5, 1.5) * label.module_px)))
        col = 255 if rng.random() < 0.8 else 0
        pts = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.line(img, *pts, col, t, cv2.LINE_AA)
        cv2.line(mask.view(np.uint8), *pts, 1, t)
    label.damage.append("scratch")


def tear(label, rng, severity=0.15):
    """A piece of the label is gone, full height. Background shows through."""
    img, mask = label.image, label.damage_mask
    a, b = _bar_span(label)
    width = severity * (b - a)
    cx = rng.uniform(a + width / 2, b - width / 2)
    top, bot = 0, img.shape[0]
    # Irregular left and right edges.
    ys = np.linspace(top, bot, 8)
    left = [(int(cx - width / 2 + rng.normal(0, width * 0.15)), int(y)) for y in ys]
    right = [(int(cx + width / 2 + rng.normal(0, width * 0.15)), int(y)) for y in ys[::-1]]
    poly = np.array(left + right, np.int32)
    fill = int(rng.choice([255, 235, 200]))
    cv2.fillPoly(img, [poly], fill, cv2.LINE_AA)
    cv2.fillPoly(mask.view(np.uint8), [poly], 1)
    label.damage.append("tear")


def occlusion(label, rng, severity=0.15):
    """Sticker, thumb or marker. Partial or full height, light or dark."""
    img, mask = label.image, label.damage_mask
    a, b = _bar_span(label)
    width = severity * (b - a)
    x = rng.uniform(a, b - width)
    full = rng.random() < 0.6
    h = label.bar_bottom - label.bar_top
    y = label.bar_top if full else int(rng.uniform(label.bar_top, label.bar_top + h * 0.5))
    y2 = label.bar_bottom + 2 if full else int(y + h * rng.uniform(0.3, 0.6))
    col = int(rng.choice([255, 240, 30]))
    cv2.rectangle(img, (int(x), y), (int(x + width), y2), col, -1)
    cv2.rectangle(mask.view(np.uint8), (int(x), y), (int(x + width), y2), 1, -1)
    label.damage.append("occlusion")


def smear(label, rng, severity=0.2):
    """Ink spread: bars grow into spaces over a region, so widths lie."""
    img, mask = label.image, label.damage_mask
    a, b = _bar_span(label)
    width = severity * (b - a)
    x = int(rng.uniform(a, b - width))
    x2 = int(x + width)
    k = max(2, int(round(label.module_px * rng.uniform(0.5, 1.0))))
    region = img[:, x:x2]
    img[:, x:x2] = cv2.erode(region, np.ones((1, k), np.uint8))
    mask[:, x:x2] = True
    label.damage.append("smear")


def fade(label, rng, severity=0.5):
    """Thermal labels in sunlight. Contrast drops, unevenly."""
    img = label.image.astype(float)
    h, w = img.shape
    ramp = np.linspace(1 - severity, 1 - severity * 0.2, w)[None, :]
    if rng.random() < 0.5:
        ramp = ramp[:, ::-1]
    img = 255 - (255 - img) * ramp
    label.image = np.clip(img, 0, 255).astype(np.uint8)
    label.damage.append("fade")


def wrinkle(label, rng, severity=0.4):
    """Folds and crumples. Crossing a fold, the paper that went into the fold
    is lost from view, so everything beyond it shifts sideways. Folds are not
    quite parallel to the bars, so the shift differs from row to row, and the
    fold itself is a shading ridge.

    `severity` is roughly the total sideways shift, in modules, divided by 3.
    """
    img = label.image.astype(np.float32)
    h, w = img.shape
    a, b = _bar_span(label)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    shade = np.zeros((h, w), np.float32)
    n = int(rng.integers(1, 4))
    for _ in range(n):
        xf = rng.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
        slope = rng.uniform(-0.15, 0.15)            # fold not parallel to bars
        width = label.module_px * rng.uniform(0.6, 2.0)
        shift = severity * 3 * label.module_px / n * rng.choice([-1, 1])
        line = xf + slope * (yy - h / 2)
        dx += shift * 0.5 * (1 + np.tanh((xx - line) / width))
        shade += rng.uniform(-60, 60) * np.exp(-((xx - line) / (1.5 * width)) ** 2)
    out = cv2.remap(img, xx - dx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    label.image = np.clip(out + shade, 0, 255).astype(np.uint8)
    label.damage_mask |= np.abs(np.gradient(dx, axis=1)) > 0.15
    label.damage.append("wrinkle")


def displaced(label, rng, severity=0.4):
    """Torn through and put back slightly wrong: the right-hand piece is moved
    sideways (leaving a gap, or overlapping and hiding modules), up or down,
    and turned a little. Every bar on both pieces is intact. A decoder that
    assumes one straight grid reads the second piece out of phase.

    `severity` scales the sideways move: up to severity * 6 modules.
    """
    img = label.image.astype(np.float32)
    h, w = img.shape
    a, b = _bar_span(label)
    xc = rng.uniform(a + 0.25 * (b - a), b - 0.25 * (b - a))
    ys = np.linspace(0, h, 9)
    edge = np.array([[xc + rng.normal(0, label.module_px), y] for y in ys], np.float32)
    left_mask = np.zeros((h, w), np.uint8)
    poly = np.vstack([[[0, 0]], edge, [[0, h]]]).astype(np.int32)
    cv2.fillPoly(left_mask, [poly], 1)

    move = severity * 6 * label.module_px * rng.uniform(0.4, 1.0) * rng.choice([-1, 1])
    lift = rng.uniform(-2, 2) * label.module_px
    turn = rng.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((float(xc), h / 2), turn, 1.0)
    M[0, 2] += move
    M[1, 2] += lift
    right = cv2.warpAffine(img, M, (w, h), borderValue=255)
    right_paper = cv2.warpAffine((1 - left_mask).astype(np.float32), M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    left_paper = left_mask.astype(np.float32)
    # The moved piece lies on top where they overlap.
    out = np.where(right_paper > 0, right, img)
    paper = np.clip(left_paper + right_paper, 0, 1)
    label.image = out.astype(np.uint8)
    label.paper = paper if label.paper is None else label.paper * paper
    gap = (paper == 0) | ((left_paper > 0) & (right_paper > 0))
    label.damage_mask |= gap
    label.damage.append("displaced")


LABEL_DAMAGE = {"scratch": scratch, "tear": tear, "occlusion": occlusion,
                "smear": smear, "fade": fade, "wrinkle": wrinkle,
                "displaced": displaced}


# ---- camera ----------------------------------------------------------------

def photograph(label, rng, tilt=0.15, rotate=10.0, glare=0.0, motion=0.0,
               blur=0.6, noise=4.0, jpeg=85, canvas=1.6):
    """Place the label in a larger frame and photograph it badly."""
    img = label.image
    h, w = img.shape
    H, W = int(h * canvas * 1.6), int(w * canvas)
    bg = rng.uniform(90, 170)
    frame = np.full((H, W), bg, np.float32)
    frame += cv2.GaussianBlur(rng.normal(0, 25, (H, W)).astype(np.float32), (0, 0), 15)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = W / 2, H / 2
    ang = np.deg2rad(rng.uniform(-rotate, rotate))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    corners = (src - [w / 2, h / 2]) @ rot.T
    jitter = rng.uniform(-tilt, tilt, (4, 2)) * [w, h] * 0.5
    dst = np.float32(corners + jitter + [cx, cy])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img.astype(np.float32), M, (W, H),
                                 flags=cv2.INTER_LINEAR, borderValue=-1)
    sheet = label.paper if label.paper is not None else np.ones_like(img, np.float32)
    paper = cv2.warpPerspective(sheet.astype(np.float32), M, (W, H),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    frame = np.where(paper > 0, warped, frame)

    if glare > 0:
        gx, gy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.35, 0.65) * H
        yy, xx = np.mgrid[0:H, 0:W]
        r = w * rng.uniform(0.08, 0.2)
        frame += glare * 255 * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))
    if motion > 0:
        k = max(3, int(motion * label.module_px * 2) | 1)
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        a = rng.uniform(0, 180)
        R = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), a, 1)
        kern = cv2.warpAffine(kern, R, (k, k))
        kern /= max(kern.sum(), 1e-6)
        frame = cv2.filter2D(frame, -1, kern)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    frame += rng.normal(0, noise, frame.shape)
    out = np.clip(frame, 0, 255).astype(np.uint8)
    if jpeg:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def random_code(rng, prefix=None):
    """A valid EAN-13 with an assigned prefix."""
    from . import gs1
    while True:
        head = prefix or "{:03d}".format(rng.integers(0, 1000))
        body = "".join(str(d) for d in rng.integers(0, 10, 12 - len(head)))
        code = ean13.complete(head + body)
        if gs1.is_assigned(code):
            return code


def sample(rng, kind, severity, code=None, module_px=None, camera=None):
    code = code or random_code(rng)
    label = render(code, module_px=module_px or rng.uniform(2.2, 4.0))
    if kind != "clean":
        LABEL_DAMAGE[kind](label, rng, severity)
    cam = dict(tilt=0.08, rotate=8.0, blur=0.5, noise=3.0, jpeg=90)
    cam.update(camera or {})
    image = photograph(label, rng, **cam)
    return image, label

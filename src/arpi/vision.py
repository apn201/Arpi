"""Pixels to a confidence map. OpenCV 5 does the image work.

    localise    where is the symbol, and which way do its bars run
    rectify     cut it out, rotated so the bars are vertical, upsampled so a
                module is several pixels wide
    scan        many scanlines, a module grid fitted per line, a vote per module

The grid fit is the step everything downstream depends on. It does not look
for bar edges and round them to whole modules, which is where naive decoders
break. It slides a continuous (offset, module width) grid over the scanline and
scores it against the 35 modules the specification fixes regardless of the
number: guards, and the first and last module of every digit. Damage that
destroys the guards still leaves dozens of digit edges to fit against.
"""
import cv2
import numpy as np

from . import ean13
from .confmap import ConfidenceMap

KNOWN = ean13.known_modules()
KNOWN_IDX = np.array(sorted(KNOWN))
KNOWN_BIT = np.array([KNOWN[i] for i in KNOWN_IDX])
MAX_RUN = 4                     # no EAN-13 element is wider than 4 modules


# ---- localise ----------------------------------------------------------------

class Region:
    def __init__(self, points, theta, score):
        self.points = points          # contour, full-resolution pixels
        self.theta = theta            # gradient direction, radians: across bars
        self.score = score


def localise(gray, work=900, max_regions=3):
    """Gradient orientation coherence. Bars give a strong gradient that points
    the same way over a wide area; text, texture and the label edge do not.
    A single long edge is coherent too, but thin, and the opening removes it."""
    h, w = gray.shape
    s = min(1.0, work / max(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    f = small.astype(np.float32)
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    sigma = max(3.0, max(small.shape) / 120)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (energy + 1e-6)
    # Gradient magnitude, not energy, and saturated early. Energy goes with
    # contrast squared, so a label faded to a third of its contrast scored a
    # ninth and fell under the threshold, half a symbol at a time.
    e = np.sqrt(energy)
    e = e / (0.25 * np.percentile(e, 99.5) + 1e-6)
    score = coherence * np.clip(e, 0, 1)

    u8 = np.clip(score * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(sigma * 2) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Connected components, not external contours. Photographed on a desk,
    # the paper's own edge is a closed loop of coherent gradient, and with
    # RETR_EXTERNAL that loop was the only region found: every code on the
    # sheet sat inside it and was dropped.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for lab in range(1, n):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < 0.002 * mask.size:
            continue
        sel = labels == lab
        ys, xs = np.nonzero(sel)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        # A symbol fills its box; a paper edge, a table edge or a cable is a
        # thin line or loop that fills almost none of it.
        if area < 0.45 * max(rw * rh, 1):
            continue
        a, b, d = jxx[sel].sum(), jyy[sel].sum(), jxy[sel].sum()
        theta = 0.5 * np.arctan2(2 * d, a - b)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        regions.append(Region(hull / s, float(theta), float(score[sel].mean() * area)))
    regions.sort(key=lambda r: -r.score)
    return _merge_fragments(regions)[:max_regions]


def regions_from_polygons(gray, polygons, work=900):
    """Regions from quadrilaterals someone else found, typically OpenCV's
    own barcode detector. Its boxes are often right even where its decoder
    then fails or misreads, so they are the natural hand-over point: OpenCV
    finds and tries, ARPI takes the box when OpenCV cannot read it.

    The bar direction is measured here, from the structure tensor inside the
    box, rather than taken from the box's corner order."""
    h, w = gray.shape
    s = min(1.0, work / max(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    f = small.astype(np.float32)
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    out = []
    for poly in polygons:
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        if len(pts) < 3 or not np.isfinite(pts).all():
            continue
        m = np.zeros(small.shape, np.uint8)
        cv2.fillPoly(m, [np.round(pts * s).astype(np.int32)], 1)
        sel = m.astype(bool)
        if sel.sum() < 20:
            continue
        a = float((gx[sel] ** 2).sum())
        b = float((gy[sel] ** 2).sum())
        d = float((gx[sel] * gy[sel]).sum())
        theta = 0.5 * np.arctan2(2 * d, a - b)
        out.append(Region(pts, float(theta), float(sel.sum())))
    return out


def _centre_inside(region, others):
    c = region.points.mean(0)
    for o in others:
        if cv2.pointPolygonTest(cv2.convexHull(o.points), (float(c[0]), float(c[1])),
                                False) >= 0:
            return True
    return False


def propose(gray, polygons=(), max_regions=30):
    """All candidate symbols in a frame: OpenCV's detections first, then any
    of ARPI's own regions that no detection already covers."""
    first = regions_from_polygons(gray, polygons)
    own = [r for r in localise(gray, max_regions=max_regions)
           if not _centre_inside(r, first)
           and not any(_centre_inside(f, [r]) for f in first)]
    return (first + own)[:max_regions]


def _merge_fragments(regions, max_angle=np.deg2rad(8)):
    """A tear or a sticker splits one symbol into two coherent regions side by
    side. Pieces that share the bar direction, overlap along the bars and sit
    next to each other along the scan line are one symbol; scanning the bigger
    piece alone throws away the digits on the other side of the damage."""
    merged, used = [], set()
    for i, r in enumerate(regions):
        if i in used:
            continue
        pts, score = r.points, r.score
        e1 = np.array([np.cos(r.theta), np.sin(r.theta)])
        e2 = np.array([-e1[1], e1[0]])
        # Compared against the original piece, never the growing union: on a
        # sheet of twelve codes the first version merged eight of them and the
        # keyboard behind the paper into one region, because each merge made
        # the union overlap the next code.
        ur, vr = r.points @ e1, r.points @ e2
        for j in range(i + 1, len(regions)):
            q = regions[j]
            if j in used:
                continue
            d = abs((q.theta - r.theta + np.pi / 2) % np.pi - np.pi / 2)
            if d > max_angle:
                continue
            ub, vb = q.points @ e1, q.points @ e2
            # Two halves of one torn code have the same bar height.
            hr, hq = np.ptp(vr), np.ptp(vb)
            if not 0.6 < hq / max(hr, 1e-6) < 1.6:
                continue
            overlap = min(vr.max(), vb.max()) - max(vr.min(), vb.min())
            if overlap < 0.6 * min(hr, hq):
                continue
            gap = max(ub.min() - ur.max(), ur.min() - ub.max())
            if gap > 0.35 * max(np.ptp(ur), np.ptp(ub)):
                continue
            # Together they must still be about one symbol wide: no wider
            # than the two pieces plus a modest gap.
            width = max(ur.max(), ub.max()) - min(ur.min(), ub.min())
            if width > 1.4 * (np.ptp(ur) + np.ptp(ub)):
                continue
            pts = np.vstack([pts, q.points])
            score += q.score
            used.add(j)
        merged.append(Region(pts, r.theta, score))
    merged.sort(key=lambda r: -r.score)
    return merged


# ---- rectify -------------------------------------------------------------------

# How far the crop extends past the localised region along the scan line, as
# a share of the region's width. Wide on purpose: a region that stopped at a
# piece of tape may be half the symbol.
CROP_PAD = 0.5


def rectify(gray, region, pad=None, vpad=0.45, target_module_px=5.0):
    """Rotate the region so the scan direction is +x. Keystone along the bars
    is left in on purpose: the per-row grid fit absorbs it, one row at a time.

    The crop is padded across the bars by `vpad` of the region height on both
    sides, because the printed digits sit outside the coherent-gradient region
    and the text reader needs them. Which side they are on is not known until
    the orientation is.

    Returns (rectified float image, affine rect->image 2x3, (top, bottom) rows
    of the localised region inside the crop).
    """
    c, s = np.cos(region.theta), np.sin(region.theta)
    e1, e2 = np.array([c, s]), np.array([-s, c])
    pts = region.points
    u, v = pts @ e1, pts @ e2
    u0, u1 = u.min(), u.max()
    v0, v1 = v.min(), v.max()
    pad = CROP_PAD if pad is None else pad
    width, height = u1 - u0, v1 - v0
    u0, u1 = u0 - pad * width, u1 + pad * width
    v0e, v1e = v0 - vpad * height, v1 + vpad * height
    # Upsample so a module is several pixels wide. Sub-pixel accuracy is then
    # a matter of the grid fit, not of interpolating between two samples.
    est_module = width / ean13.MODULES
    scale = max(1.0, target_module_px / max(est_module, 0.5))
    W = int(round((u1 - u0) * scale))
    H = int(round((v1e - v0e) * scale))
    # rectified (x, y) -> image: origin + x/scale * e1 + y/scale * e2
    origin = u0 * e1 + v0e * e2
    A = np.array([[e1[0] / scale, e2[0] / scale, origin[0]],
                  [e1[1] / scale, e2[1] / scale, origin[1]]], np.float32)
    out = cv2.warpAffine(gray.astype(np.float32), A, (W, H),
                         flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    top = int(round((v0 - v0e) * scale))
    bottom = int(round((v1 - v0e) * scale))
    return out, A, (top, bottom)


# ---- scan ----------------------------------------------------------------------

def _positions(a, b, k, u):
    """Pixel position of module boundary u (0..95) on a scanline.

    x(u) = a + b u + k b u (u - 95) / 95

    a is where module 0 starts, b the mean module width, and k bends the grid
    so module width runs from b(1-k) at one end to b(1+k) at the other. That is
    what a label tilted away from the camera does to a scanline, and a linear
    grid cannot follow it: on a 15 degree tilt it is a whole module out by the
    far guard. a, b and k broadcast against each other.
    """
    a, b, k = (np.asarray(x, float)[..., None] for x in (a, b, k))
    return a + b * u + k * b * u * (u - ean13.MODULES) / ean13.MODULES


def _box_means(profile, a, b, k, modules, frac=0.5):
    """Mean of `profile` over the central `frac` of each module. Sub-pixel, by
    integrating a cumulative sum rather than reading the nearest pixel."""
    cum = np.concatenate([[0.0], np.cumsum(profile)])
    xs = np.arange(len(cum), dtype=float)
    lo_edge = _positions(a, b, k, modules.astype(float))
    hi_edge = _positions(a, b, k, modules + 1.0)
    centre, half = (lo_edge + hi_edge) / 2, (hi_edge - lo_edge) * frac / 2
    lo = np.clip(centre - half, 0, len(profile))
    hi = np.clip(centre + half, 0, len(profile))
    integ = np.interp(hi, xs, cum) - np.interp(lo, xs, cum)
    return integ / np.maximum(hi - lo, 1e-6)


_PATTERNS_LEFT = np.array([[1 if c == "1" else -1 for c in code]
                           for code in ean13.L_CODES + ean13.G_CODES], float)
_PATTERNS_RIGHT = np.array([[1 if c == "1" else -1 for c in code]
                            for code in ean13.R_CODES], float)
_GUARD_IDX = np.array(sorted(ean13.GUARDS))
_GUARD_SIGN = np.array([1 if ean13.GUARDS[i] else -1 for i in _GUARD_IDX], float)
_ALL = np.arange(ean13.MODULES)
_KB = KNOWN_IDX[KNOWN_BIT == 1]
_KW = KNOWN_IDX[KNOWN_BIT == 0]


def _trend(v, idx):
    """Least-squares straight line through v at the known modules `idx`,
    evaluated at every module. Fading and uneven light are gradients, so the
    black and white levels are lines, not constants."""
    x = idx - idx.mean()
    y = v[..., idx]
    ym = y.mean(-1, keepdims=True)
    slope = ((y - ym) * x).sum(-1, keepdims=True) / (x * x).sum()
    return ym + slope * (_ALL - idx.mean())


def _normalise(v):
    """Module means -> signed darkness in [-1, 1], +1 = bar."""
    black, white = _trend(v, _KB), _trend(v, _KW)
    span = white - black
    sgn = np.clip(2 * (white - v) / np.maximum(span, 1e-3) - 1, -1, 1)
    return np.where(span > 0, sgn, 0.0)


def _fit_score(profile, a, b, k, off=None):
    """How decodable the scanline is under this grid: for every digit, how
    well its seven modules match the best legal pattern, plus the guards.

    A template of fixed modules alone is too periodic to fit against - shift
    the grid by one digit and the guards-plus-digit-edges template still
    half-matches. Asking whether every digit looks like *some* real digit is
    much sharper, and it still needs no knowledge of the number.
    """
    mods = _ALL if off is None else _ALL + off
    sgn = _normalise(_box_means(profile, a, b, k, mods))
    total = (sgn[..., _GUARD_IDX] * _GUARD_SIGN).sum(-1)
    for pos in range(12):
        st = ean13.digit_start(pos)
        pats = _PATTERNS_LEFT if pos < 6 else _PATTERNS_RIGHT
        total = total + (sgn[..., st:st + 7] @ pats.T).max(-1)
    return total / ean13.MODULES


def _grid_search(profile, a_range, b_range, k_range, off=None):
    A, B, K = np.meshgrid(a_range, b_range, k_range, indexing="ij")
    sc = _fit_score(profile, A, B, K, off)
    i = np.unravel_index(np.argmax(sc), sc.shape)
    return float(A[i]), float(B[i]), float(K[i]), float(sc[i])


K_COARSE = np.array([-0.12, 0.0, 0.12])
COARSE_DOWNSAMPLE = True
RESCAN_FULL_BELOW = 0.93
WIDTH_STEP = 0.025         # coarse module-width step, as a share of the guess


def module_from_runs(profile):
    """Module width from the visible bars alone. Every bar and space is 1-4
    modules wide, so the narrow runs measure the module directly, however
    much of the symbol is hidden. Returns None if there are too few runs.

    This exists because of masking tape. The localiser's box ended where the
    tape lowered the contrast, the grid fit squeezed all 95 modules into the
    55 it could see, and the result was 41 wrong modules read with full
    confidence."""
    W = len(profile)
    win = max(5, W // 25)
    p = profile.astype(np.float32).reshape(1, -1)
    kern = np.ones((1, win), np.uint8)
    white = cv2.GaussianBlur(cv2.dilate(p, kern), (0, 0), win / 3).ravel()
    black = cv2.GaussianBlur(cv2.erode(p, kern), (0, 0), win / 3).ravel()
    span = white - black
    strong = span > 0.4 * span.max()
    dark = profile < (white + black) / 2
    runs, start = [], None
    for i in range(1, W):
        if dark[i] != dark[i - 1] or not strong[i]:
            if start is not None and strong[i - 1]:
                runs.append(i - start)
            start = i if strong[i] else None
    runs = np.array(runs[1:-1], float)            # ends are cut by the edge
    runs = runs[runs >= 1]
    if len(runs) < 12:
        return None
    b0 = np.percentile(runs, 20)
    if b0 <= 0:
        return None
    mult = np.clip(np.round(runs / b0), 1, 4)
    return float(runs.sum() / mult.sum())


def fit_grid(profile, pad=None, hint=None):
    """Search every offset, width and bend the rectified crop allows, coarse
    then fine. Two independent guesses at the module width are searched
    around: the crop width (the localiser's box was the symbol, padded by
    `pad` each side) and the run lengths of the visible bars. The first is
    right when the whole symbol was found, the second when part of it is
    hidden; the decodability score picks between them."""
    if hint is not None:
        # A rescan of the same symbol on slightly different rows: the grid
        # barely moves, so only look near it.
        a, b, k = hint
        a, b, k, _ = _grid_search(profile, a + np.arange(-2, 2.01, 0.25) * b,
                                  b * np.arange(0.97, 1.031, 0.01),
                                  k + np.array([-0.04, 0.0, 0.04]))
    else:
        pad = CROP_PAD if pad is None else pad
        W = len(profile)
        nominal = W / (1 + 2 * pad) / ean13.MODULES
        # Widths in 2.5% steps, then the best few refined. Searching every 1%
        # was most of the scan's time on Lambda's slower cores, and a 2.5%
        # step is still close enough for the refinement to find the peak.
        widths = list(nominal * np.arange(0.70, 1.15, WIDTH_STEP))
        from_runs = module_from_runs(profile)
        if from_runs:
            widths += list(from_runs * np.arange(0.85, 1.16, WIDTH_STEP))
        # Coarse pass on a half-resolution profile: the crop is upsampled to
        # about five pixels a module, and two and a half is plenty to find
        # roughly where the grid sits. The fine pass is at full resolution.
        f = 2.0 if COARSE_DOWNSAMPLE and min(widths) >= 4 else 1.0
        small = profile if f == 1 else cv2.resize(
            profile.astype(np.float32).reshape(1, -1),
            (int(round(W / f)), 1), interpolation=cv2.INTER_AREA).ravel()
        hits = []
        for b in widths:
            if ean13.MODULES * b > W:
                continue
            bs = b / f
            a_range = np.arange(0, max(1.0, len(small) - ean13.MODULES * bs), 0.25 * bs)
            hits.append(_grid_search(small, a_range, np.array([bs]), K_COARSE))
        if not hits:
            hits = [(0.0, nominal / f, 0.0, -1e9)]
        # Refine the three best coarse hits on the full profile: between two
        # coarse width steps, the true peak can sit under either neighbour.
        hits.sort(key=lambda h: -h[3])
        best = (0.0, nominal, 0.0, -1e9)
        for ha, hb, hk, _ in hits[:3]:
            hit = _grid_search(profile,
                               ha * f + np.arange(-1.0, 1.01, 0.125) * hb * f,
                               hb * f * np.arange(0.975, 1.0251, 0.005),
                               hk + np.array([-0.06, 0.0, 0.06]))
            if hit[3] > best[3]:
                best = hit
        a, b, k = best[0], best[1], best[2]
    a, b, k, _ = _grid_search(profile,
                              a + np.arange(-0.4, 0.401, 0.04) * b,
                              b * np.arange(0.985, 1.0151, 0.0015),
                              k + np.arange(-0.06, 0.061, 0.02))
    # A last, fine pass. Coarsening the one above for speed cost a torn,
    # tilted label its clean read: at the far guard a hundredth of a module
    # in width is a whole module in position.
    return _grid_search(profile,
                        a + np.arange(-0.04, 0.041, 0.01) * b,
                        b * np.arange(0.9985, 1.00151, 0.0003),
                        k + np.arange(-0.01, 0.011, 0.005))


# ---- piecewise warp: wrinkles, folds, torn pieces put back wrong -------------
#
# One grid, however bent, assumes the paper is one flat piece. A fold hides
# the paper that went into it and shifts everything beyond it; a torn piece
# put back is shifted, and turned, as a whole. Both look like this along a
# scanline: segments of the symbol, each intact, each displaced sideways by
# its own amount.
#
# So each of the 15 segments (3 guards, 12 digits) gets its own shift, and the
# shifts are chosen together by dynamic programming. Neighbours moving
# smoothly is cheap (a wrinkle); one abrupt jump costs a fixed amount (a
# tear); not moving at all is preferred.
#
# The risk is overfitting. Give a digit freedom to slide and it will usually
# find some legal pattern by luck, and the bars it then reports are confident
# and wrong. Three guards against that: the shift range stays under half a
# digit, so a segment cannot slide onto its neighbour's pattern; a warp is
# kept only if it beats the plain grid by WARP_MARGIN, which is set so clean
# labels never warp; and every row's warp is pulled toward the consensus.

SEGMENTS = ([(0, 3, "guard")]
            + [(ean13.digit_start(p), 7, "left") for p in range(6)]
            + [(45, 5, "guard")]
            + [(ean13.digit_start(p), 7, "right") for p in range(6, 12)]
            + [(92, 3, "guard")])
_SEG_OF = np.concatenate([[i] * n for i, (_, n, _) in enumerate(SEGMENTS)])
SHIFTS = np.arange(-3.0, 3.001, 0.125)          # modules; < half a digit
WARP_LAMBDA = 1.5      # cost per module of smooth change between neighbours
WARP_JUMP = 5.0        # cost of one abrupt change, wherever and however big
WARP_STAY = 0.15       # cost per module of shift, so no evidence means no shift
WARP_MARGIN = 4.0      # a warp must beat the plain grid by this much
WARP_QUALITY = 0.6     # ...counting only segments that end up looking real


def _envelopes(profile, b):
    """Local white and black levels along the scanline. A fold is a shading
    ridge and a crumple is a dozen of them, so one straight trend for the
    whole line, which is enough for fading, is not enough here."""
    win = max(3, int(round(8 * b)))
    p = profile.astype(np.float32).reshape(1, -1)
    kern = np.ones((1, win), np.uint8)
    white = cv2.GaussianBlur(cv2.dilate(p, kern), (0, 0), win / 2).ravel()
    black = cv2.GaussianBlur(cv2.erode(p, kern), (0, 0), win / 2).ravel()
    return white, black


def _segment_scores(profile, a, b, k):
    """(15 segments, len(SHIFTS)): how well each segment, moved by each
    shift, looks like a legal guard or digit. Summed over its modules."""
    mods = _ALL[None, :] + SHIFTS[:, None]                    # (S, 95)
    v = _box_means(profile, a, b, k, mods)
    centres = _positions(a, b, k, mods + 0.5)
    white_env, black_env = _envelopes(profile, b)
    xs = np.arange(len(profile), dtype=float)
    white = np.interp(centres, xs, white_env)
    black = np.interp(centres, xs, black_env)
    span = white - black
    sgn = np.clip(2 * (white - v) / np.maximum(span, 1e-3) - 1, -1, 1)
    sgn = np.where(span > 4, sgn, 0.0)
    out = np.zeros((len(SEGMENTS), len(SHIFTS)))
    for i, (st, n, kind) in enumerate(SEGMENTS):
        seg = sgn[:, st:st + n]
        if kind == "guard":
            out[i] = seg @ np.array([1 if ean13.GUARDS[st + j] else -1
                                     for j in range(n)], float)
        else:
            pats = _PATTERNS_LEFT if kind == "left" else _PATTERNS_RIGHT
            out[i] = (seg @ pats.T).max(-1)
    return out


def _viterbi(scores, prior=None, pull=0.0):
    """Best shift per segment. `prior` (15,) pulls each segment toward a
    given path, used to keep per-row warps near the consensus."""
    S = len(SHIFTS)
    d = np.abs(SHIFTS[:, None] - SHIFTS[None, :])
    trans = np.minimum(WARP_LAMBDA * d, WARP_JUMP)
    unary = scores - WARP_STAY * np.abs(SHIFTS)[None, :]
    if prior is not None:
        unary = unary - pull * np.abs(SHIFTS[None, :] - np.asarray(prior)[:, None])
    acc = unary[0].copy()
    back = np.zeros((len(scores), S), int)
    for i in range(1, len(scores)):
        cand = acc[:, None] - trans                  # [from, to]
        back[i] = np.argmax(cand, axis=0)
        acc = cand[back[i], np.arange(S)] + unary[i]
    path = [int(np.argmax(acc))]
    for i in range(len(scores) - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path = path[::-1]
    return SHIFTS[path], float(acc.max())


def fit_warp(profile, a, b, k, prior=None, pull=0.0):
    """Returns (per-module offsets (95,), per-segment shifts (15,), gain).
    Offsets are all zero unless the warp beat the plain grid by WARP_MARGIN."""
    scores = _segment_scores(profile, a, b, k)
    shifts, _ = _viterbi(scores, prior, pull)
    zero = int(np.argmin(np.abs(SHIFTS)))
    idx = np.searchsorted(SHIFTS, shifts - 1e-9)
    moved = scores[np.arange(len(SEGMENTS)), idx]
    plain = scores[:, zero]
    sizes = np.array([n for _, n, _ in SEGMENTS], float)
    # Gain counts only where the shifted segment now looks like a real guard
    # or digit. Sliding a segment across a sticker or a tear until the junk
    # looks marginally less like junk is the overfitting this has to refuse:
    # measured before this rule, occlusions reached gains of 17 while clean
    # labels never passed 3.4.
    real = moved >= WARP_QUALITY * sizes
    gain = float((moved - plain)[real].sum())
    if gain < WARP_MARGIN:
        return np.zeros(ean13.MODULES), np.zeros(len(SEGMENTS)), float(gain)
    return shifts[_SEG_OF], shifts, float(gain)


SEGMENT_NAMES = (["start guard"] + ["digit {}".format(i) for i in range(2, 8)]
                 + ["centre guard"] + ["digit {}".format(i) for i in range(8, 14)]
                 + ["end guard"])


def describe_warp(shifts):
    """Plain words for the agent and the HUD: where the paper moved."""
    shifts = np.asarray(shifts, float)
    out = []
    for i, st in enumerate(np.diff(shifts)):
        if abs(st) >= 1.0:
            out.append("break between {} and {}, {:+.1f} modules".format(
                SEGMENT_NAMES[i], SEGMENT_NAMES[i + 1], st))
    if not out and np.ptp(shifts) >= 0.5:
        out.append("smooth distortion, {:.1f} modules end to end".format(np.ptp(shifts)))
    return out


def _run_flags(bits):
    """Modules inside a run wider than any EAN-13 element allows."""
    flags = np.zeros(len(bits), bool)
    start = 0
    for i in range(1, len(bits) + 1):
        if i == len(bits) or bits[i] != bits[start]:
            if i - start > MAX_RUN:
                flags[start:i] = True
            start = i
    return flags


def scan(rect, band=None, rows=32, hint=None):
    """Confidence map from the rows in `band` (top, bottom). Returns
    (ConfidenceMap, (a, b, k)) or (None, None)."""
    H, W = rect.shape
    top, bottom = band or (0, H)
    span = bottom - top
    idx = np.arange(top + int(span * 0.04), top + int(span * 0.96))
    if len(idx) < 2:
        return None, None
    bands = np.array_split(idx, min(rows, len(idx)))
    profiles = np.stack([rect[i].mean(axis=0) for i in bands if len(i)])

    median = np.median(profiles, axis=0)
    a, b, k, gscore = fit_grid(median, hint=hint)
    off_g, shifts_g, gain_g = fit_warp(median, a, b, k)
    if shifts_g.any():
        gscore = max(gscore, float(_fit_score(median, a, b, k, off_g)))

    per_row_p, weights, contrasts, row_warps = [], [], [], 0
    for prof in profiles:
        ra, rb, rk, rs = _grid_search(prof,
                                      a + np.arange(-0.5, 0.51, 0.1) * b,
                                      b * np.arange(0.98, 1.0201, 0.004),
                                      k + np.array([-0.02, 0.0, 0.02]), off_g)
        if rs < 0.5 * gscore:
            ra, rb, rk = a, b, k          # a damaged row keeps the global grid
        # A fold that is not parallel to the bars moves each row by a
        # different amount, so each row may refine the consensus warp, pulled
        # toward it so one noisy row cannot invent its own tear.
        off = off_g
        if shifts_g.any():
            off_r, sh_r, _ = fit_warp(prof, ra, rb, rk, prior=shifts_g, pull=2.0)
            if sh_r.any():
                off, row_warps = off_r, row_warps + 1
        v = _box_means(prof, ra, rb, rk, _ALL + off)
        black, white = _trend(v, _KB), _trend(v, _KW)
        contrast = float(np.median(white - black))
        if contrast <= 1:
            continue
        p = np.clip((white - v) / np.maximum(white - black, 1), 0, 1)
        agree = np.mean((p[KNOWN_IDX] > 0.5) == (KNOWN_BIT == 1))
        per_row_p.append(p)
        weights.append(contrast * agree ** 2)
        contrasts.append(contrast)

    if not per_row_p:
        return None, None
    P = np.stack(per_row_p)
    w = np.array(weights)
    w = w / w.max()
    w[w < 0.25] = 0                       # glare, text rows, torn-off rows
    if w.sum() == 0:
        w[:] = 1

    # Weighted median per module, and how many rows agree with it.
    order = np.argsort(P, axis=0)
    Ps = np.take_along_axis(P, order, 0)
    Ws = w[order]
    cw = np.cumsum(Ws, 0) / Ws.sum(0)
    med = Ps[np.argmax(cw >= 0.5, axis=0), _ALL]
    side = (P > 0.5) == (med > 0.5)[None, :]
    agree = (w[:, None] * side).sum(0) / w.sum()
    conf = np.clip(2 * agree - 1, 0, 1)

    # Run-length sanity: a run wider than 4 modules is impossible inside the
    # symbol, so whatever produced it (sticker, tear, blot) is not evidence.
    run_flags = _run_flags(med > 0.5)
    conf[run_flags] *= 0.15

    glare = float(np.mean(rect[top:bottom] >= 250))
    sharp = float(np.mean(np.abs(med[KNOWN_IDX] - 0.5)) * 2)
    return ConfidenceMap(
        p_bar=med.tolist(), confidence=conf.tolist(),
        module_px=float(b), rows_used=int((w > 0).sum()),
        diagnostics={
            "contrast": round(float(np.median(contrasts)) / 255, 3),
            "glare_fraction": round(glare, 3),
            "sharpness": round(sharp, 3),
            "rows_rejected": round(float(np.mean(w == 0)), 3),
            "impossible_runs": int(run_flags.sum()),
            "grid_fit": round(float(gscore), 3),
            "grid_bend": round(float(k), 3),
            "warp": [round(float(x), 2) for x in shifts_g],
            "warp_gain": round(float(gain_g), 2),
            "warp_rows": row_warps,
            "warp_notes": describe_warp(shifts_g),
            "left_quality": round(float(conf[3:45].mean()), 3),
            "right_quality": round(float(conf[50:92].mean()), 3),
        }), (a, b, k)


def barness(rect, grid, p_bar):
    """Per row of the crop, how well it matches the consensus bar pattern.
    Bars score near 1, the printed digits and the paper near 0. This is how
    the reader finds where the bars actually end and the text begins, rather
    than trusting the localiser's box."""
    a, b, k = grid
    lo = _positions(a, b, k, _ALL.astype(float))
    hi = _positions(a, b, k, _ALL + 1.0)
    centre, half = (lo + hi) / 2, (hi - lo) / 4
    W = rect.shape[1]
    left = np.clip(centre - half, 0, W)
    right = np.clip(centre + half, 0, W)
    cum = np.concatenate([np.zeros((rect.shape[0], 1)), np.cumsum(rect, 1)], 1)
    xs = np.arange(W + 1, dtype=float)
    # Linear interpolation of every row's cumulative sum at the module edges,
    # done for all rows at once.
    i0 = np.clip(np.floor(left).astype(int), 0, W - 1)
    i1 = np.clip(np.floor(right).astype(int), 0, W - 1)
    f0, f1 = left - i0, right - i1
    c0 = cum[:, i0] * (1 - f0) + cum[:, i0 + 1] * f0
    c1 = cum[:, i1] * (1 - f1) + cum[:, i1 + 1] * f1
    vals = (c1 - c0) / np.maximum(right - left, 1e-6)
    vals = vals - vals.mean(1, keepdims=True)
    sign = 1 - 2 * np.asarray(p_bar)              # +1 space, -1 bar
    sign = sign - sign.mean()
    num = (vals * sign).sum(1)
    den = np.sqrt((vals ** 2).sum(1) * (sign ** 2).sum()) + 1e-6
    return num / den


def _bar_rows(score, threshold=0.5):
    """Longest run of rows that look like bars."""
    good = score > threshold
    best, start = (0, 0), None
    for i, g in enumerate(np.r_[good, False]):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


class Reading:
    """Everything one region of one frame produced. The candidate engine needs
    only `cmap`; the text reader needs the crop, grid and bar rows; the HUD
    needs `affine` to draw it all back onto the camera image."""

    def __init__(self, region, rect, affine, grid, bars, cmap):
        self.region, self.rect, self.affine = region, rect, affine
        self.grid, self.bars, self.cmap = grid, bars, cmap
        self.reversed = False

    def flip(self):
        """Turn the reading 180 degrees: crop, grid, bar rows and map. After
        this the bars read left to right and the digits sit underneath."""
        H, W = self.rect.shape
        a, b, k = self.grid
        self.rect = cv2.rotate(self.rect, cv2.ROTATE_180)
        self.grid = (W - a - ean13.MODULES * b, b, -k)
        self.bars = (H - self.bars[1], H - self.bars[0])
        # rect point (x, y) now came from (W - x, H - y) of the old crop
        F = np.array([[-1, 0, W], [0, -1, H], [0, 0, 1]], np.float64)
        A = np.vstack([self.affine.astype(np.float64), [0, 0, 1]]) @ F
        self.affine = A[:2].astype(np.float32)
        self.cmap = self.cmap.reversed()
        self.cmap.orientation = "forward"
        # Segment order reverses with the symbol, and a shift toward higher
        # module numbers becomes a shift toward lower ones.
        warp = self.cmap.diagnostics.get("warp")
        if warp:
            self.cmap.diagnostics["warp"] = [-x for x in warp[::-1]]
            self.cmap.diagnostics["warp_notes"] = describe_warp(
                self.cmap.diagnostics["warp"])
        self.reversed = not self.reversed
        return self

    def to_image(self, pts):
        """Crop coordinates (N, 2) -> camera image coordinates."""
        pts = np.asarray(pts, np.float64)
        return pts @ self.affine[:, :2].T.astype(np.float64) + self.affine[:, 2]


def read_frame(gray, max_regions=2, regions=None):
    """Every plausible reading in the frame, best region first. `regions`
    skips localisation, for callers that already have them."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    out = []
    for region in (regions if regions is not None
                   else localise(gray, max_regions=max_regions)):
        rect, A, band = rectify(gray, region)
        if rect.shape[0] < 8 or rect.shape[1] < ean13.MODULES:
            continue
        cmap, grid = scan(rect, band)
        if cmap is None:
            continue
        # The localiser's box is a guess at where the bars are. The grid fit
        # gives a better one: rows that match the consensus pattern. Rescan
        # on those if they differ much, which also keeps text rows out.
        bars = _bar_rows(barness(rect, grid, cmap.p_bar))
        if bars[1] - bars[0] > 8 and (abs(bars[0] - band[0]) + abs(bars[1] - band[1])
                                      > 0.1 * (band[1] - band[0])):
            cm2, g2 = scan(rect, bars, hint=grid)
            # The hint carries the first scan's grid, and the first scan ran
            # on the localiser's band, which on a torn label can include a
            # flap. When the hinted fit is mediocre, search afresh on the
            # clean rows too and keep the better. Clean labels fit above
            # 0.95 and never pay for this.
            if cm2 is None or cm2.diagnostics["grid_fit"] < RESCAN_FULL_BELOW:
                cm3, g3 = scan(rect, bars)
                if cm3 is not None and (cm2 is None or cm3.diagnostics["grid_fit"]
                                        > cm2.diagnostics["grid_fit"]):
                    cm2, g2 = cm3, g3
            if cm2 is not None and cm2.diagnostics["grid_fit"] >=                     cmap.diagnostics["grid_fit"] - 0.02:
                cmap, grid = cm2, g2
        else:
            bars = band if bars[1] - bars[0] <= 8 else bars
        out.append(Reading(region, rect, A, grid, bars, cmap))
    return out


def read_maps(gray, max_regions=2):
    """Confidence maps only. Kept for callers that want nothing else."""
    return [r.cmap for r in read_frame(gray, max_regions)]

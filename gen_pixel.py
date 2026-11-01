import argparse
import colorsys
import numpy as np

SIZE = 32
GRIDS  = [16, 20, 24, 28, 32]
GRID_P = [0.10, 0.20, 0.30, 0.25, 0.15]

BG, BODY, ACCENT, EYE, EDGE = 0, 1, 2, 3, 4


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v)) * 255


def mirror(half, n):
    """Левая половина (n, ceil(n/2)) -> полная симметричная (n, n)."""
    right = half[:, ::-1][:, n % 2:]
    return np.concatenate([half, right], axis=1)


def neighbors(m):
    p = np.pad(m.astype(int), 1)
    return sum(p[i:i + m.shape[0], j:j + m.shape[1]]
               for i in range(3) for j in range(3)) - m.astype(int)


def make_mask(rng, n):
    half_w = (n + 1) // 2
    for _ in range(50):
        # 1) грубая форма (силуэт) на сетке в 2-4 раза меньше
        k = int(rng.choice([2, 3, 4])) if n >= 20 else 1
        cn = max(4, n // k)
        chw = (cn + 1) // 2
        coarse = rng.random((cn, chw)) < rng.uniform(0.40, 0.65)
        for _ in range(rng.integers(1, 3)):
            nb = neighbors(coarse)
            coarse = (nb >= 4) | (coarse & (nb >= 2))
        # растягиваем до размера (n, half_w)
        ys = (np.arange(n) * cn // n)
        xs = (np.arange(half_w) * chw // half_w)
        m = coarse[np.ix_(ys, xs)]

        # 2) мелкие детали: вырезы и выступы на границе, случайные дырки внутри
        noise = rng.random((n, half_w))
        edge = neighbors(m) < 8
        m = np.where(edge & (noise < 0.18), ~m, m)           # зазубренный край
        m = m & ~((noise > 0.93) & ~edge)                     # мелкие дырки внутри
        full = mirror(m, n)
        if 0.2 < full.mean() < 0.65 and full[:, n // 2].any():
            return m, full
    return m, full


def make_sprite(rng, n):
    half_w = (n + 1) // 2
    half, mask = make_mask(rng, n)

    idx = np.zeros((n, n), dtype=np.int8)
    idx[mask] = BODY

    if rng.random() < 0.8:
        acc_half = (rng.random((n, half_w)) < rng.uniform(0.1, 0.4)) & half
        idx[mirror(acc_half, n)] = ACCENT

    if rng.random() < 0.45:
        p = np.pad(mask, 1)
        inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        idx[mask & ~inner & (idx == BODY)] = EDGE

    rows = np.where(mask.any(1))[0]
    top, bot = rows.min(), rows.max()
    r_lo, r_hi = top, top + max(1, (bot - top) // 2)
    cand = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(n // 2)
            if mask[r, c] and mask[r, n - 1 - c]]
    eyes = []
    if cand and rng.random() < 0.92:
        r, c = cand[rng.integers(len(cand))]
        eyes = [(r, c), (r, n - 1 - c)]
        es = max(1, n // 12)
        for rr, cc in eyes:
            idx[rr:rr + es, cc:cc + es] = EYE
        if rng.random() < 0.35 and r + 2 <= bot:
            rr = min(bot, r + rng.integers(2, 4))
            for cc in range(n // 2 - 1, n - n // 2 + 1):
                if 0 <= cc < n and mask[rr, cc]:
                    idx[rr, cc] = EYE
    return idx


def make_palette(rng):
    h = rng.random()
    s = rng.uniform(0.45, 0.95)
    v = rng.uniform(0.72, 1.0)
    body = hsv(h, s, v)
    if rng.random() < 0.07:
        g = rng.uniform(200, 255)
        body = np.array([g, g, g])
    accent = hsv(h + rng.choice([0.5, 0.33, -0.33, 0.12, -0.12]),
                 rng.uniform(0.4, 0.95), rng.uniform(0.75, 1.0))
    edge = body * rng.uniform(0.45, 0.7)
    eye_dark = rng.random() < 0.8
    eye = np.array([10, 10, 14]) if eye_dark else np.array([245, 245, 245])
    bg_h = h + rng.uniform(-0.15, 0.15) if rng.random() < 0.6 else rng.random()
    bg = hsv(bg_h, rng.uniform(0.25, 0.7), rng.uniform(0.04, 0.13))
    return bg, body, accent, edge, eye


def make_background(rng, bg):
    img = np.empty((SIZE, SIZE, 3))
    img[:] = bg
    if rng.random() < 0.6:
        g = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.5), SIZE)
        img *= g[:, None, None]
    if rng.random() < 0.3:
        h0 = rng.integers(26, 30)
        for y in range(h0, SIZE, 2):
            for x in range(0, SIZE, 2):
                img[y:y + 2, x:x + 2] *= rng.uniform(0.8, 1.35)
    if rng.random() < 0.5:
        for _ in range(rng.integers(1, 7)):
            x, y = rng.integers(0, SIZE, 2)
            img[y, x] = bg * 3 + rng.uniform(20, 70)
    return img


def render(rng):
    n = int(rng.choice(GRIDS, p=GRID_P))
    s = SIZE // n
    sprite = make_sprite(rng, n)
    bg, body, accent, edge, eye = make_palette(rng)
    img = make_background(rng, bg)

    cols = np.stack([bg, body, accent, edge, eye])
    rgb = cols[sprite]
    if rng.random() < 0.5:
        shade = np.linspace(1.12, 0.82, n)[:, None, None]
        m = np.isin(sprite, [BODY, ACCENT, EDGE])[..., None]
        rgb = np.where(m, rgb * shade, rgb)

    big = np.repeat(np.repeat(rgb, s, 0), s, 1)
    big_mask = np.repeat(np.repeat(sprite != BG, s, 0), s, 1)
    side = n * s
    slack = SIZE - side
    ox = slack // 2
    oy = slack // 2 + (rng.integers(-1, 2) if slack >= 2 else 0)
    oy = int(np.clip(oy, 0, slack))
    region = img[oy:oy + side, ox:ox + side]
    region[big_mask] = big[big_mask]
    return np.clip(img, 0, 255).astype(np.uint8)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([render(rng) for _ in range(n)])


def save_preview(images, path="preview.png", cols=20, rows=20, scale=3):
    from PIL import Image
    k = cols * rows
    grid = images[:k].reshape(rows, cols, SIZE, SIZE, 3)
    grid = np.pad(grid, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
    s = SIZE + 2
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * s, cols * s, 3)
    im = Image.fromarray(grid).resize((cols * s * scale, rows * s * scale),
                                      Image.NEAREST)
    im.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pixel_sprites.npz")
    ap.add_argument("--preview", default="preview.png")
    a = ap.parse_args()

    X = generate(a.n, a.seed)
    np.savez_compressed(a.out, images=X)
    save_preview(X, a.preview)
    print(f"saved {X.shape} {X.dtype} -> {a.out}, preview -> {a.preview}")

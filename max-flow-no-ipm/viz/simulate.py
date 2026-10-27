"""Animate the discrete potential process from Section 3 of no-ipm.tex.

Each step finds the cut S maximizing b = f(in(S)) / f(out(S)) by brute force
over all subsets, and raises the potential of S by eps / f(cut(S)). The process
stops once every cut has b <= 2, i.e. the induced flow is a 2-pseudo-circulation.

Run: viz/.venv/bin/python viz/simulate.py [--seed N ...]  (see --help)
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.spatial import Delaunay

SEED = 6
N_INNER = 13  # vertices besides s and t
U = 10  # capacities are integers in [1, U]; anchors have capacity U
EPS = 0.05  # step size: eta = EPS / f(cut(S))
# BETA = 2.0
BETA = 1.1
P_BOTH, P_FWD = 0.0, 0.6  # per neighboring pair: both directions, rightward only
MAX_FRAMES = 150
OUT_DIR = Path(__file__).parent
# Oblique view for the lifted renders: the layout lies on a floor seen from an
# angle (depth z = layout y is sheared by KX and squashed by KY), and each vertex
# is raised off the floor by LIFT times its potential.
KX, KY = 0.25, 0.7
ANCHOR_RAD = 0.32
LIFT_FRAC = 0.75  # the highest final vertex sits LIFT_FRAC * layout height up

# Light chart surface and text inks.
SURFACE = "#ffffff"
INK = "#1f1f1e"
INK_MUTED = "#6b6a66"
# Blue -> purple -> red ramp for arc flow / capacity.
CMAP = LinearSegmentedColormap.from_list(
    "flow", ["#1c5cab", "#8a4a9e", "#c23b3a"]
)
CUT_COLOR = "#1c5cab"  # vertices in the cut being raised; others are INK


def make_graph(rng):
    """Random strongly connected digraph on a jittered layout, s left, t right."""
    while True:
        pts = [(0.0, 1.0)]  # s
        while len(pts) < N_INNER + 1:
            p = (rng.uniform(0.8, 5.2), rng.uniform(0.0, 2.0))
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.6 for q in pts):
                pts.append(p)
        pts.append((6.0, 1.0))  # t
        pos = np.array(pts)
        n = len(pos)
        s, t = 0, n - 1

        pairs = set()
        for simplex in Delaunay(pos).simplices:
            for i in range(3):
                a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
                if np.linalg.norm(pos[a] - pos[b]) < 2.0:
                    pairs.add((a, b))

        arcs = []
        for a, b in sorted(pairs):
            # Bias arcs rightward so that there is a real s-t flow to route.
            fwd, bwd = (a, b), (b, a)
            if pos[b][0] < pos[a][0]:
                fwd, bwd = bwd, fwd
            r = rng.random()
            if r < P_BOTH:
                arcs += [fwd, bwd]
            elif r < P_BOTH + P_FWD:
                arcs.append(fwd)
            else:
                arcs.append(bwd)

        g = nx.DiGraph(arcs)
        g.add_edge(t, s)  # the anchors
        if g.number_of_nodes() == n and nx.is_strongly_connected(g):
            cap = rng.integers(1, U + 1, size=len(arcs))
            return pos, s, t, np.array(arcs), cap


def induced_flow(y, tails, heads, cap):
    return 1.0 / (np.maximum(0.0, y[heads] - y[tails]) + 1.0 / cap)


def simulate(n, s, t, arcs, cap):
    m = len(arcs)
    # Arc m is the bundle of m anchors (t, s), each of capacity U.
    tails = np.append(arcs[:, 0], t)
    heads = np.append(arcs[:, 1], s)
    caps = np.append(cap, U).astype(float)
    mult = np.append(np.ones(m), m)

    # Every nonempty proper subset S, as a boolean row.
    masks = np.arange(1, 2**n - 1)
    member = (masks[:, None] >> np.arange(n)) & 1 == 1
    out_cut = (member[:, tails] & ~member[:, heads]).astype(float)
    in_cut = (~member[:, tails] & member[:, heads]).astype(float)

    y = np.zeros(n)
    history = []
    while True:
        f = induced_flow(y, tails, heads, caps)
        fin, fout = in_cut @ (f * mult), out_cut @ (f * mult)
        b = fin / fout
        k = int(np.argmax(b))
        history.append(dict(y=y.copy(), frac=f / caps, S=member[k], b=b[k]))
        if b[k] <= BETA:
            return history
        y = y + member[k] * EPS / (fin[k] + fout[k])


def draw_arc(ax, p, q, color, width, rad=0.12, z=1):
    ax.add_patch(
        FancyArrowPatch(
            p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=9, color=color, lw=width, shrinkA=9, shrinkB=9,
            zorder=z,
        )
    )


def project(pos, y, lift):
    """Screen coordinates of the layout pos with vertex potentials y."""
    if lift is None:
        return pos
    z = pos[:, 1] - pos[:, 1].mean()
    return np.column_stack([pos[:, 0] + KX * z, KY * z + lift * y])


def render(history, pos, s, t, arcs, cap, frame_ids, lifted=True, shadows=True):
    """Yield one figure per frame: the lifted graph above the plot of max b."""
    n = len(pos)
    ymax = max(history[-1]["y"].max(), 1e-9)
    lift = LIFT_FRAC * np.ptp(pos[:, 1]) / ymax if lifted else None
    # Potentials only increase, so the final frame bounds every frame.
    lo = project(pos, np.zeros(n), lift)
    hi = project(pos, history[-1]["y"], lift)
    xlim = (min(lo[:, 0].min(), hi[:, 0].min()) - 0.4,
            max(lo[:, 0].max(), hi[:, 0].max()) + 0.4)
    # The anchor arc's apex (arc3 is a quadratic Bezier) can top everything.
    apex = (hi[s, 1] + hi[t, 1]) / 2 + 0.5 * ANCHOR_RAD * abs(hi[t, 0] - hi[s, 0])
    ylim = (lo[:, 1].min() - 0.45, max(hi[:, 1].max(), apex) + 0.25)
    # Back-to-front drawing order for vertices: farther (larger depth) first.
    depth_rank = np.argsort(np.argsort(-pos[:, 1]))
    frac_norm = Normalize(0, 1)
    bs = np.array([h["b"] for h in history])
    B_TICKS = [x for x in (1, 2, 5, 10, 20, 50, 100, 200) if x <= bs.max() * 1.5]
    widths = 0.8 + 0.5 * cap

    for i in frame_ids:
        h = history[i]
        fig = plt.figure(figsize=(8, 4.8), dpi=100, facecolor=SURFACE)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.12,
                              left=0.08, right=0.98, top=0.99, bottom=0.1)
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor(SURFACE)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")

        p = project(pos, h["y"], lift)
        if shadows:
            floor = project(pos, np.zeros(n), lift)
            for v in range(n):
                ax.plot([floor[v, 0], p[v, 0]], [floor[v, 1], p[v, 1]],
                        color="#b9b7b1", lw=0.9, ls=(0, (2, 2)), zorder=0.5)
            ax.scatter(floor[:, 0], floor[:, 1], s=40, c="#d6d4cf", zorder=0.4,
                       linewidths=0)

        for e, (u, v) in enumerate(arcs):
            draw_arc(ax, p[u], p[v], CMAP(frac_norm(h["frac"][e])), widths[e])
        # The anchor bundle, drawn once as a wide arc over the top of the graph.
        draw_arc(ax, p[t], p[s], CMAP(frac_norm(h["frac"][-1])), 3.5, rad=ANCHOR_RAD)

        S = h["S"] if h["b"] > BETA else np.zeros(n, bool)
        for v in range(n):
            ax.scatter(*p[v], s=180, color=CUT_COLOR if S[v] else INK,
                       zorder=3 + 0.01 * depth_rank[v], edgecolors=SURFACE,
                       linewidths=2)
        for v, label, dx in ((s, "s", -0.25), (t, "t", 0.25)):
            ax.text(p[v][0] + dx, p[v][1], label, ha="center", va="center",
                    fontsize=11, color=INK, style="italic", zorder=5)

        bx = fig.add_subplot(gs[1])
        bx.set_facecolor(SURFACE)
        bx.plot(np.arange(len(bs)), bs, color=INK, lw=1.5)
        bx.axhline(BETA, color=INK_MUTED, lw=1, ls=(0, (4, 3)))
        bx.text(len(bs) * 0.005, BETA * 1.12, "b = 2", ha="left", va="bottom",
                fontsize=8, color=INK_MUTED)
        bx.axvline(i, color="#c23b3a", lw=1.5)
        bx.set_yscale("log")
        bx.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(B_TICKS))
        bx.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bx.set_xlim(0, (len(bs) - 1) * 1.01)
        bx.set_xlabel("step", fontsize=8, color=INK_MUTED)
        bx.set_ylabel("max b", fontsize=8, color=INK_MUTED)
        bx.tick_params(labelsize=8, colors=INK_MUTED)
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            bx.spines[side].set_color("#c9c8c4")

        yield fig
        plt.close(fig)


def write_mp4(out, figs, n_frames, fps=12.5, dpi=150):
    """Encode frames with ffmpeg, matching the GIF's timing: hold the first frame
    for about 0.6s and the last for about 2.5s."""
    proc = None
    for k, fig in enumerate(figs):
        fig.set_dpi(dpi)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if proc is None:
            h, w = frame.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                 "-movflags", "+faststart", str(out)],
                stdin=subprocess.PIPE)
        repeats = 8 if k == 0 else 31 if k == n_frames - 1 else 1
        for _ in range(repeats):
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")


def main():
    global SEED, U, P_BOTH, P_FWD
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cap", type=int, default=U, help="max capacity U")
    ap.add_argument("--p-both", type=float, default=P_BOTH)
    ap.add_argument("--p-fwd", type=float, default=P_FWD)
    ap.add_argument("--out", default="process.gif",
                    help="a .gif, an .mp4, or a .pdf with one page per frame")
    ap.add_argument("--still", action="store_true",
                    help="render only the final frame")
    args = ap.parse_args()
    SEED, U, P_BOTH, P_FWD = args.seed, args.cap, args.p_both, args.p_fwd

    rng = np.random.default_rng(SEED)
    pos, s, t, arcs, cap = make_graph(rng)
    history = simulate(len(pos), s, t, arcs, cap)
    print(f"{len(pos)} vertices, {len(arcs)} arcs (+ anchors), "
          f"{len(history) - 1} steps, max potential {history[-1]['y'].max():.3f}")

    last = len(history) - 1
    frame_ids = sorted(set(np.linspace(0, last, min(MAX_FRAMES, last + 1)).round().astype(int)))
    if args.still:
        frame_ids = [last]
    figs = render(history, pos, s, t, arcs, cap, frame_ids)
    out = OUT_DIR / args.out
    if out.suffix == ".pdf":
        with PdfPages(out) as pdf:
            for fig in figs:
                pdf.savefig(fig)
    elif out.suffix == ".mp4":
        write_mp4(out, figs, len(frame_ids))
    else:
        frames = []
        for fig in figs:
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
                          .convert("RGB"))
        durations = [600] + [80] * (len(frames) - 2) + [2500]
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=durations[: len(frames)], loop=0, optimize=True)
    print(f"wrote {args.out} ({len(frame_ids)} frames)")


if __name__ == "__main__":
    main()

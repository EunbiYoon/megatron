"""Export every TensorBoard run under ./runs into one multi-page PDF.

Layout:
  1. Summary table (one row per run: last step, finished or not).
  2. Key metrics, one section per game: all runs of that game overlaid per plot
     (val win rates, format/overlong losses, response length, KL/entropy/grad, lr, step time).
  3. Appendix: every other scalar tag, overlaying all runs that logged it.

Usage (login node is fine, it only reads event files and plots):
  python scripts/tb_to_pdf.py                      # -> reports/tb_report_<timestamp>.pdf
  python scripts/tb_to_pdf.py --runs runs --out my.pdf --smooth 0.6
"""
import argparse
import fnmatch
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MAX_STEP = 199  # max_steps: 200 -> last step index 199

# Short names for runs we already talk about; anything else is labeled by its path.
KNOWN_LABELS = {
    "tictactoe_selfplay/20260924-001348": "m1b (ttt, lr1e-6, no min_lr)",
    "tictactoe_selfplay_quickcheck/20260924-001348": "m2b (ttt quickcheck)",
    "tictactoe_selfplay_opt1/20261004-004234": "opt1 (ttt, lr5e-6)",
    "tictactoe_selfplay_opt2/20261006-004511": "opt2 (ttt, lr1e-6, min_lr1e-7)",
    "kuhn_poker_selfplay/20261001-152723": "k1b (kuhn)",
    "kuhn_poker_selfplay_quickcheck/20261002-091306": "k2b (kuhn quickcheck)",
}

# Key metrics in display order. {G} is replaced by the run's training env name (TicTacToe, KuhnPoker).
KEY_SECTIONS = [
    ("Validation (vs built-in opponents)", ["val/score/mean", "val/score/max", "val/score/min", "val/env/*"]),
    ("Self-play outcome (train)", [
        "env/{G}/success", "env/{G}/winner", "env/{G}/draw",
        "env/{G}/player_0_success", "env/{G}/player_1_success",
    ]),
    ("Losses from format / length", [
        "env/{G}/player_*_lose_for_wrong_format",
        "env/{G}/player_*_lose_for_overlong_response",
        "env/{G}/player_*_lose_for_overlong_sequence",
    ]),
    ("Game length & response length", [
        "env/{G}/num_actions", "env/{G}/response_length", "env/{G}/response_length_turn_*",
        "env/{G}/response_length_player_*_turn_*",
        "tokens/response_length/*", "tokens/non_prompt_length/*", "tokens/prompt_length/mean",
    ]),
    ("Reward & entropy", ["critic/score/*", "critic/entropy/mean"]),
    ("Policy update (PPO / KL)", [
        "actor/lr", "actor/approxkl", "actor/kl_loss", "actor/pg_loss", "actor/clipfrac",
        "actor/ppo_ratio_clipfrac", "actor/ppo_ratio_high_clipfrac", "actor/ppo_ratio_low_clipfrac",
        "actor/ratio_max", "actor/ratio_mean", "actor/ratio_min", "actor_train/grad_norm",
    ]),
    ("Step time breakdown (minutes)", [
        "derived/step_time_total_min", "time/rollout", "time/ref_log_probs_values_reward",
        "time/old_log_probs_values", "time/actor_train/train_step/total",
        "time/actor_train/model_update/total",
    ]),
]
STEP_TIME_PARTS = [
    "time/rollout", "time/ref_log_probs_values_reward", "time/old_log_probs_values",
    "time/actor_train/train_step/total", "time/actor_train/model_update/total",
]


def find_runs(root):
    """A run is <root>/<group>/<timestamp>/tensorboard containing event files directly."""
    runs = []
    for group in sorted(os.listdir(root)):
        gdir = os.path.join(root, group)
        if not os.path.isdir(gdir):
            continue
        for ts in sorted(os.listdir(gdir)):
            tb = os.path.join(gdir, ts, "tensorboard")
            if os.path.isdir(tb) and any(f.startswith("events.out.tfevents") for f in os.listdir(tb)):
                runs.append((f"{group}/{ts}", tb))
    return runs


def load_run(key, tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        evs = ea.Scalars(tag)
        data[tag] = ([e.step for e in evs], [e.value for e in evs], [e.wall_time for e in evs])
    # Derived: total step time in minutes from the main phases.
    parts = [dict(zip(data[t][0], data[t][1])) for t in STEP_TIME_PARTS if t in data]
    if parts:
        steps = sorted(set.intersection(*[set(p) for p in parts]))
        data["derived/step_time_total_min"] = (steps, [sum(p[s] for p in parts) / 60 for s in steps], [])
    for t in STEP_TIME_PARTS:  # show phase times in minutes too
        if t in data:
            s, v, w = data[t]
            data[t] = (s, [x / 60 for x in v], w)
    game = None
    for tag in data:
        m = re.match(r"env/([^/]+)/", tag)
        if m:
            game = m.group(1)
            break
    return {"key": key, "label": KNOWN_LABELS.get(key, key), "game": game or "unknown", "data": data}


def ema(values, alpha):
    if alpha <= 0 or len(values) < 3:
        return None
    out, prev = [], values[0]
    for v in values:
        prev = alpha * prev + (1 - alpha) * v
        out.append(prev)
    return out


def match_tags(patterns, tags, game):
    out = []
    for p in patterns:
        p = p.replace("{G}", game)
        for t in sorted(tags):
            if fnmatch.fnmatchcase(t, p) and t not in out:
                out.append(t)
    return out


def plot_pages(pdf, title, tags, runs, smooth, per_page=6):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_of = {r["key"]: colors[i % len(colors)] for i, r in enumerate(runs)}
    for start in range(0, len(tags), per_page):
        chunk = tags[start:start + per_page]
        fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
        fig.suptitle(f"{title}  ({start // per_page + 1}/{(len(tags) - 1) // per_page + 1})", fontsize=12)
        for ax, tag in zip(axes.flat, chunk):
            for r in runs:
                if tag not in r["data"]:
                    continue
                steps, vals, _ = r["data"][tag]
                c = color_of[r["key"]]
                # val metrics are logged every eval_steps -> draw as markers, no smoothing
                gaps = sorted(b - a for a, b in zip(steps, steps[1:]))
                sparse = len(steps) < 3 or gaps[len(gaps) // 2] > 1
                sm = None if sparse else ema(vals, smooth)
                if sm is None:
                    ax.plot(steps, vals, color=c, lw=1.2, marker="o" if sparse else None, ms=2.5, label=r["label"])
                else:
                    ax.plot(steps, vals, color=c, lw=0.6, alpha=0.3)
                    ax.plot(steps, sm, color=c, lw=1.4, label=r["label"])
            ax.set_title(tag, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            ax.set_xlabel("step", fontsize=7)
        for ax in list(axes.flat)[len(chunk):]:
            ax.axis("off")
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)


def summary_page(pdf, runs, root):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    rows = []
    for r in runs:
        steps = [s for t, (st, _, _) in r["data"].items() for s in st]
        last = max(steps) if steps else -1
        wall = [w for _, (_, _, ws) in r["data"].items() for w in ws]
        last_t = datetime.fromtimestamp(max(wall)).strftime("%Y-%m-%d %H:%M") if wall else "-"
        status = "finished" if last >= MAX_STEP else f"stopped/running at {last}"
        rows.append([r["label"], r["key"], r["game"], str(last), last_t, status])
    table = ax.table(cellText=rows, colLabels=["run", "path (under runs/)", "game", "last step", "last log", "status"],
                     loc="upper center", cellLoc="left", bbox=None)
    widths = [0.22, 0.36, 0.08, 0.06, 0.12, 0.16]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(widths[col])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title(f"TensorBoard report: {os.path.abspath(root)}\ngenerated {datetime.now():%Y-%m-%d %H:%M}",
                 fontsize=12, pad=20)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=f"reports/tb_report_{datetime.now():%Y%m%d-%H%M}.pdf")
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA weight like TensorBoard's slider (0 = off)")
    args = ap.parse_args()

    runs = [load_run(k, d) for k, d in find_runs(args.runs)]
    if not runs:
        raise SystemExit(f"no runs with tensorboard events under {args.runs}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    shown = set()
    with PdfPages(args.out) as pdf:
        summary_page(pdf, runs, args.runs)
        for game in sorted({r["game"] for r in runs}):
            game_runs = [r for r in runs if r["game"] == game]
            tags = set().union(*[r["data"].keys() for r in game_runs])
            for title, patterns in KEY_SECTIONS:
                sel = match_tags(patterns, tags, game)
                if sel:
                    plot_pages(pdf, f"[{game}] {title}", sel, game_runs, args.smooth)
                    shown.update(sel)
        all_tags = sorted(set().union(*[r["data"].keys() for r in runs]) - shown)
        for prefix in sorted({t.split("/")[0] for t in all_tags}):
            sel = [t for t in all_tags if t.split("/")[0] == prefix]
            plot_pages(pdf, f"[Appendix] {prefix}/*", sel, runs, args.smooth)
    print(f"wrote {args.out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()

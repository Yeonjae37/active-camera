"""3D knee-flexion accuracy across the views of a rendered sweep (pan_tilt, rail or orbit).

GT: flexion of the SMPL hip-knee-ankle joints, one value per frame for every view. Prediction: NLF SMPL-24
joints3d, same formula. Writes per_image_<mode>.csv, summary.json and figures to <experiment>/analysis/ (next to the capture folder).
All "best" values use GT (post-hoc upper bounds) except best_fixed_train / phase_locked_train, which pick on the
train frames and are scored on the test frames. Rail sweeps also get the motion-constrained oracle
(rest-to-rest moves under configs/rail_limits.json, dt = 1 / motion fps).

    python -m src.simulation.analyze output/02_rail_viewpoint/capture
    python -m src.simulation.analyze output/01_pantilt_framing/capture --train 79 121 --test 139 163
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import LEGS, ROOT, gait_phase, gt_flexion, gt_identity_error, knee_flexion, load_sweep, project
from .estimators import infer_sweep
from .planning import constrained_oracle, phase_consistency, policies, switch_stats, viewpoint_variance

MODES = ["raw", "crop"]
OCCLUSION_GAP_M = -0.15  # rendered depth - joint depth below this: a surface lies in front of the joint
VIEW = {"pan_tilt": ("view", (0, 0)), "rail": ("rail_position_m", 0.0), "orbit": ("orbit_azimuth_deg", 90)}


def table(caps, j3, gt, phase):
    rows = []
    for c, j in zip(caps, j3):
        ok = bool(np.isfinite(j).all())
        uv, _ = project(c, c["gt"])
        r = {"file": c["file"], "frame": c["frame"], "view": (c["pan"], c["tilt"]),
             **{k: c[k] for k in ("rail_position_m", "orbit_azimuth_deg", "view_azimuth_walk_deg", "camera_to_pelvis_m")},
             "body_height_px": float(np.ptp(uv[:, 1]) * 1.15), "phase": phase[c["frame"]][0], "cycle": phase[c["frame"]][1],
             "identity_error_gt": gt_identity_error(j, c["gt"]) if ok else np.nan}
        for s, (_, k, a) in LEGS.items():
            p = knee_flexion(j, s) if ok else np.nan
            r.update({f"{s}_flex_gt_deg": gt[c["frame"]][s], f"{s}_flex_pred_deg": p,
                      f"{s}_flex_err_deg": abs(p - gt[c["frame"]][s]),
                      f"{s}_occluded": bool(np.nanmin([c["gap"][k], c["gap"][a], np.inf]) < OCCLUSION_GAP_M)})
        r["mean_flex_err_deg"] = (r["left_flex_err_deg"] + r["right_flex_err_deg"]) / 2
        rows.append(r)
    return pd.DataFrame(rows)


def factors(df, key):
    """Within-frame relations (frame means removed) and leg-level occlusion / flexion effects."""
    dm = df[["mean_flex_err_deg", "view_azimuth_walk_deg", "camera_to_pelvis_m", "body_height_px"]].astype(float)
    dm = dm - dm.groupby(df.frame).transform("mean")
    out = {f"within_frame_spearman_{k}": float(spearmanr(dm[k], dm.mean_flex_err_deg).statistic)
           for k in dm.columns[1:] if dm[k].std() > 1e-9}  # NaN std (field not recorded) is skipped too
    if key == "rail_position_m" and dm.notna().all().all():  # azimuth and distance co-vary along the rail: separate them
        beta = np.linalg.lstsq(dm[["view_azimuth_walk_deg", "camera_to_pelvis_m"]], dm.mean_flex_err_deg, rcond=None)[0]
        out["within_frame_ols"] = {"deg_per_10deg_azimuth": float(beta[0] * 10), "deg_per_m_distance": float(beta[1])}
    for s in LEGS:
        o = df[f"{s}_occluded"]
        out[f"{s}_occlusion"] = {"rate": float(o.mean()), "err_occluded": float(df.loc[o, f"{s}_flex_err_deg"].mean()),
                                 "err_visible": float(df.loc[~o, f"{s}_flex_err_deg"].mean())}
    gt_err = pd.concat([df[[f"{s}_flex_gt_deg", f"{s}_flex_err_deg"]].set_axis(["gt", "err"], axis=1) for s in LEGS])
    out["error_vs_gt_flexion_spearman"] = float(spearmanr(gt_err["gt"], gt_err["err"]).statistic)
    return out


def analyse(df, key, ref, kind, train, test, fps, limits):
    piv = df.pivot_table(index="frame", columns=key, values="mean_flex_err_deg", aggfunc="first")
    views, E, fr = list(piv.columns), piv.to_numpy(float), piv.index.to_numpy()
    if np.isnan(E).any():
        raise ValueError("missing estimates")
    first = df.drop_duplicates("frame").set_index("frame").loc[fr]
    phase, cycle = first.phase.to_numpy(), first.cycle.to_numpy()
    tr, te = (fr >= train[0]) & (fr <= train[1]), (fr >= test[0]) & (fr <= test[1])
    numeric = kind != "pan_tilt"
    out = {"views": [str(v) for v in views], "viewpoint_variance": viewpoint_variance(E),
           "phase_consistency_between_cycles": phase_consistency(E, phase, cycle)}
    for name, ev in [("all_frames", np.ones(len(fr), bool)), ("test_frames", te)]:
        pol, predicted = policies(E, views, views.index(ref), tr, ev, phase, fps, numeric)
        if kind == "rail":
            pos, dt = np.array(views, float), 1 / fps
            nom = limits["nominal"]
            err, traj = constrained_oracle(E[ev], pos, nom["max_velocity_m_s"], nom["max_acceleration_m_s2"], dt)
            pol["constrained_oracle_nominal"] = {"err": err, **nom, **switch_stats(pos[traj], fps), "trajectory": pos[traj].tolist()}
            pol["constrained_sweep"] = []
            for v in limits["sweep"]["max_velocity_m_s"]:
                for a in limits["sweep"]["max_acceleration_m_s2"]:
                    err, traj = constrained_oracle(E[ev], pos, v, a, dt)
                    locked = None if np.isnan(predicted).any() else constrained_oracle(predicted, pos, v, a, dt)[1]
                    pol["constrained_sweep"].append({
                        "max_velocity_m_s": v, "max_acceleration_m_s2": a, "err": err,
                        "phase_locked_train_err": None if locked is None else float(E[ev][np.arange(ev.sum()), locked].mean()),
                        **switch_stats(pos[traj], fps)})
            assert abs(constrained_oracle(E[ev], pos, None, None, dt)[0] - pol["framewise_oracle"]["err"]) < 1e-9
        if numeric:
            pol["framewise_oracle"]["trajectory"] = np.asarray(views, float)[E[ev].argmin(1)].tolist()
        out[name] = pol
    if numeric:
        best = views[int(E.mean(0).argmin())]
        gain = piv[best] - piv.min(axis=1)
        at_best = df[df[key] == best].set_index("frame").loc[fr]
        flex = at_best[["left_flex_gt_deg", "right_flex_gt_deg"]].max(axis=1)
        occ = at_best[["left_occluded", "right_occluded"]].any(axis=1)
        hi = flex >= flex.quantile(0.75)
        out["oracle_gain_vs_best_fixed"] = {
            "spearman_vs_max_flexion": float(spearmanr(flex, gain).statistic),
            "top_quartile_flexion": float(gain[hi].mean()), "other_frames": float(gain[~hi].mean()),
            "occluded_frames": float(gain[occ].mean()) if occ.any() else None, "visible_frames": float(gain[~occ].mean())}
        out["factors"] = factors(df, key)
    out["by_view"] = df.groupby(key)[["left_flex_err_deg", "right_flex_err_deg", "mean_flex_err_deg"]].mean().round(3).reset_index().astype({key: str}).to_dict("records")
    return out


def plot_views(df, key, kind, res, path, title):
    if kind == "pan_tilt":
        cell = df.groupby("view").mean_flex_err_deg.mean()
        pans, tilts = sorted({v[0] for v in cell.index}), sorted({v[1] for v in cell.index}, reverse=True)
        M = np.array([[cell[(p, t)] for p in pans] for t in tilts])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="magma_r", vmin=0)
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > M.max() / 2 else "black")
        ax.add_patch(plt.Rectangle((pans.index(0) - 0.5, tilts.index(0) - 0.5), 1, 1, fill=False, ec="red", lw=2))
        ax.set_xticks(range(len(pans)), pans)
        ax.set_yticks(range(len(tilts)), tilts)
        ax.set(xlabel="pan offset (deg)", ylabel="tilt offset (deg)")
        fig.colorbar(im, label="|3D knee flexion error| (deg)")
    else:
        cols = ["left_flex_err_deg", "right_flex_err_deg", "mean_flex_err_deg"]
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1.2, 1.2]})
        g = df.drop_duplicates("frame").set_index("frame")
        axes[0].plot(g.index, g.left_flex_gt_deg, color="tab:cyan", label="left GT")
        axes[0].plot(g.index, g.right_flex_gt_deg, color="tab:orange", label="right GT")
        axes[0].set_ylabel("GT flexion (deg)")
        axes[0].legend(fontsize=8)
        vmax = np.nanpercentile(df[cols], 99)
        views = sorted(df[key].unique())
        for ax, col in zip(axes[1:], cols):
            piv = df.pivot_table(index=key, columns="frame", values=col)
            im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=vmax, origin="lower",
                           extent=(piv.columns[0] - 0.5, piv.columns[-1] + 0.5, -0.5, len(views) - 0.5))
            ax.set_yticks(range(len(views)), views)
            ax.set_ylabel(key)
            ax.set_title(col, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="deg")
        a = res["all_frames"]
        y = lambda traj: [views.index(v) for v in traj]
        axes[-1].plot(g.index, y(a["framewise_oracle"]["trajectory"]), "w.", ms=3, label="frame-wise oracle")
        if "constrained_oracle_nominal" in a:
            axes[-1].plot(g.index, y(a["constrained_oracle_nominal"]["trajectory"]), "c-", lw=1.5, label="constrained oracle (nominal)")
        axes[-1].legend(fontsize=8, facecolor="#444444", labelcolor="white")
        axes[-1].set_xlabel("motion frame")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--out", help="default <sweep>/../analysis")
    ap.add_argument("--train", nargs=2, type=int, default=[79, 171], help="frames (cycles 1-2)")
    ap.add_argument("--test", nargs=2, type=int, default=[172, 220], help="frames (cycle 3)")
    ap.add_argument("--limits", default=ROOT / "configs/rail_limits.json")
    args = ap.parse_args()
    meta, caps = load_sweep(args.sweep)
    kind = meta.get("experiment", "pan_tilt")
    key, ref = VIEW[kind]
    out = caps[0]["path"].parent.parent / "analysis" if args.out is None else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fps = float(meta["fps"])
    frames = sorted({c["frame"] for c in caps})
    gt, (phase, strikes) = gt_flexion(frames), gait_phase(fps)
    limits = json.loads(open(args.limits).read())
    summary = {"sweep": args.sweep, "kind": kind, "frames": [frames[0], frames[-1], len(frames)], "views": len(caps) // len(frames),
               "train": args.train, "test": args.test, "left_heel_strikes": strikes, "limits": limits if kind == "rail" else None}
    for mode in MODES:
        j3, _ = infer_sweep(caps, mode, out / f"nlf_joints3d_{mode}.npz")
        df = table(caps, j3, gt, phase)
        df.to_csv(out / f"per_image_{mode}.csv", index=False)
        res = analyse(df, key, ref, kind, args.train, args.test, fps, limits)
        res["detected"] = f"{int(np.isfinite(j3).all((1, 2)).sum())}/{len(caps)}"
        res["identity_errors_gt"] = int(df.identity_error_gt.sum())
        summary[mode] = res
        plot_views(df, key, kind, res, out / f"views_{mode}.png", f"{mode}: 3D knee flexion error ({kind})")
        a, t = res["all_frames"], res["test_frames"]
        print(f"[{mode}] " + " | ".join(f"{k} {a[k]['err']:.2f}/{t[k]['err']:.2f}" for k in a if isinstance(a[k], dict) and isinstance(t[k], dict)) + "  (all/test)")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("->", out)


if __name__ == "__main__":
    main()

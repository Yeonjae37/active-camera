"""NLF 3D human pose estimator (SMPL-24 joints, same joint definition as the GT) and cached inference.

Model: models/nlf/nlf_l_multi_0.3.2.torchscript (github.com/isarandi/nlf releases, non-commercial research).
detect_smpl_batched(uint8 (1,3,H,W), intrinsic_matrix (1,3,3)) -> boxes (N,5) xywh+score, joints3d (N,24,3)
camera frame (x right, y down, z forward, mm), plus pose/betas/vertices and joint_uncertainties.
"""

import cv2
import numpy as np

from .common import ROOT, iou, project, rgb

MODEL = ROOT / "models/nlf/nlf_l_multi_0.3.2.torchscript"
CROP_SIZE = 640


class NLF:
    def __init__(self, path=MODEL, device="cuda"):
        import torch
        import torchvision  # noqa: F401  registers ops the TorchScript model needs
        self.torch, self.device = torch, device
        self.model = torch.jit.load(str(path)).to(device).eval()

    def predict(self, image, K):
        t = self.torch
        img = t.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(self.device)
        with t.inference_mode(), t.device(self.device):
            pred = self.model.detect_smpl_batched(img, intrinsic_matrix=t.as_tensor(np.float32(K))[None].to(self.device),
                                                  model_name="smpl")
        boxes, j3 = pred["boxes"][0].cpu().numpy(), pred["joints3d"][0].cpu().numpy()
        return [{"box": np.r_[b[:2], b[:2] + b[2:4]], "score": float(b[4]), "joints3d": j} for b, j in zip(boxes, j3)]


def infer(model, cap, mode):
    """Person matching the projected GT box. raw: full image; crop: square GT-box crop resized to CROP_SIZE
    with K adjusted (uniform scale + shift = same camera centre, K' = A K)."""
    image = rgb(cap["path"])
    h, w = image.shape[:2]
    uv, z = project(cap, cap["gt"])
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not inside.any():
        return None
    pad = 0.15 * np.ptp(uv[:, 1]) * 1.15
    box = np.r_[uv[inside].min(0) - pad, uv[inside].max(0) + pad]
    if mode == "raw":
        people = model.predict(image, cap["K"])
    else:
        lo, hi = uv.min(0) - pad, uv.max(0) + pad
        s = CROP_SIZE / (hi - lo).max()
        A = np.array([[s, 0, CROP_SIZE / 2 - s * (lo[0] + hi[0]) / 2], [0, s, CROP_SIZE / 2 - s * (lo[1] + hi[1]) / 2]])
        crop = cv2.warpAffine(image, A, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
        people = model.predict(crop, np.vstack([A, [0, 0, 1]]) @ cap["K"])
        Ai = cv2.invertAffineTransform(A)
        for p in people:
            p["box"] = (p["box"].reshape(2, 2) @ Ai[:, :2].T + Ai[:, 2]).reshape(-1)
    best = max(people, key=lambda p: iou(p["box"], box), default=None)
    return best if best is not None and iou(best["box"], box) > 0.1 else None


def infer_sweep(caps, mode, cache):
    """joints3d (N,24,3) and detection score per capture, cached in an npz keyed by the file list."""
    files = [c["file"] for c in caps]
    if cache.exists():
        z = np.load(cache)
        if list(z["files"]) == files:
            return z["joints3d"].astype(float), z["score"]
    model = NLF()
    j3, score = np.full((len(caps), 24, 3), np.nan, np.float32), np.full(len(caps), np.nan, np.float32)
    for i, cap in enumerate(caps):
        p = infer(model, cap, mode)
        if p is not None:
            j3[i], score[i] = p["joints3d"], p["score"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, files=np.array(files), joints3d=j3, score=score)
    return j3.astype(float), score

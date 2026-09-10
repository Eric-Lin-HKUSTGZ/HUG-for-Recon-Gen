"""Render HUG predictions as PNG — pure Python, no Viser/plotly/kaleido needed.

Usage:
    python render_predictions.py [--dataset-path data/hug_bench/] [--output-dir prediction_images/]
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# MANO skeleton + faces
# ---------------------------------------------------------------------------
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]

MANO_FACES = np.load(
    Path(__file__).resolve().parent / "assets" / "mano_rhand_mesh_faces.npy"
)  # (1552, 3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((224, 224, 3), dtype=np.uint8)


def _decode_depth(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _project(points_3d, K):
    proj = points_3d @ K.T
    return proj[:, :2] / (proj[:, 2:3] + 1e-8)


def _rotation_matrix(azim_deg, elev_deg):
    """Rotation around hand center for viewing."""
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([[np.cos(az), 0, np.sin(az)],
                    [0, 1, 0],
                    [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el), np.cos(el)]])
    return Ry @ Rx


def _render_hand_3d(mesh_verts, landmarks, faces, view_name, size=500):
    """Render hand as wireframe mesh + skeleton using perspective projection.

    Centers the view on the hand, not the camera origin.
    """
    # Determine viewpoint
    views = {
        "front":  (0, 0),          # look along +Z (camera forward)
        "side":   (90, 0),         # look along -X
        "top":    (0, 90),         # look along -Y
        "angled": (-45, 25),       # 3/4 view
    }
    azim, elev = views.get(view_name, (-45, 25))
    R = _rotation_matrix(azim, elev)

    # Center of hand
    hand_center = mesh_verts.mean(axis=0)  # (3,)
    hand_radius = np.linalg.norm(mesh_verts - hand_center, axis=1).max()

    # Rotate vertices and landmarks around hand center
    verts_centered = (mesh_verts - hand_center) @ R.T
    lm_centered = (landmarks - hand_center) @ R.T

    # Perspective projection from a camera at distance
    focal = 2.0 * hand_radius  # camera distance from hand center
    # Camera looks along -Z of rotated space, positioned at z = +focal
    z_shift = focal - verts_centered[:, 2]
    verts_proj = verts_centered[:, :2] / (verts_centered[:, 2:3] - focal) * (-focal)
    lm_proj = lm_centered[:, :2] / (lm_centered[:, 2:3] - focal) * (-focal)

    # Scale to fit image
    # Map the [-r, r] range to [margin, size-margin]
    margin = size * 0.08
    r = hand_radius * 1.3
    scale = (size - 2 * margin) / (2 * r)

    def _to_screen(p):
        u = margin + (p[:, 0] + r) * scale
        v = margin + (r - p[:, 1]) * scale  # flip Y for image coords
        return u, v

    vu, vv = _to_screen(verts_proj)
    lu, lv = _to_screen(lm_proj)

    # Depth for shading (from rotated Z, larger Z = closer to viewer)
    depths = verts_centered[:, 2]
    d_min, d_max = depths.min(), depths.max()
    d_range = d_max - d_min if d_max > d_min else 1.0
    d_norm = (depths - d_min) / d_range  # 0=far, 1=near

    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # --- Draw mesh faces as filled triangles, back-to-front for painter's algo ---
    face_depths = np.array([depths[face].mean() for face in faces])
    face_order = np.argsort(face_depths)  # far → near

    for fi in face_order:
        i0, i1, i2 = faces[fi]
        # Triangle corners in screen coords
        pu = [vu[i0], vu[i1], vu[i2]]
        pv = [vv[i0], vv[i1], vv[i2]]
        # Skip if completely outside
        if all(p < 0 for p in pu) or all(p > size for p in pu):
            continue
        if all(p < 0 for p in pv) or all(p > size for p in pv):
            continue
        # Shading based on depth
        fd = (face_depths[fi] - d_min) / d_range
        shade = int(100 + 155 * fd)  # 100 (far) → 255 (near)
        # Hand skin tone: more pinkish near, more gray far
        fill = (int(shade * 0.95), int(shade * 0.75), int(shade * 0.65))
        outline = (int(shade * 0.7), int(shade * 0.55), int(shade * 0.48))
        try:
            draw.polygon([(pu[0], pv[0]), (pu[1], pv[1]), (pu[2], pv[2])],
                         fill=fill, outline=outline)
        except Exception:
            pass

    # --- Draw skeleton on top ---
    for i, j in CONNECTIONS:
        draw.line([(lu[i], lv[i]), (lu[j], lv[j])], fill=(0, 180, 220), width=3)

    # --- Draw joints ---
    for i in range(21):
        u, v = lu[i], lv[i]
        r_j = max(3, int(hand_radius * scale * 0.06))
        draw.ellipse([u - r_j, v - r_j, u + r_j, v + r_j],
                     fill=(0, 255, 255), outline=(0, 100, 100), width=1)

    # --- Label ---
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Background for label
    label_w, label_h = 80, 24
    draw.rectangle([0, 0, label_w, label_h], fill=(50, 50, 50, 180))
    draw.text((8, 3), view_name.upper(), fill="white", font=font)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dataset_path: Path = Path("data/hug_bench/"),
    output_dir: Path = Path("prediction_images/"),
) -> None:
    pred_dir = dataset_path / "grasp_pred"
    if not pred_dir.is_dir():
        print(f"No predictions at {pred_dir}")
        sys.exit(1)

    pkl_files = sorted(pred_dir.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} prediction(s)")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            pred_data = pickle.load(f)

        rgb = _decode_jpeg(pred_data.get("image", b""))
        depth_uint16 = _decode_depth(pred_data.get("depth", b""))
        cam = pred_data.get("camera", {})
        K = np.asarray(cam.get("K", np.eye(3)), dtype=np.float64)
        grasp = pred_data.get("grasp", {})
        lm3d = np.asarray(grasp.get("landmarks_3d", np.zeros((21, 3))))
        verts = np.asarray(grasp.get("mesh_vertices", np.zeros((778, 3))))
        T_wrist = np.asarray(grasp.get("T_camera_wrist", np.eye(4)))
        stem = pkl_path.relative_to(pred_dir).with_suffix("").as_posix().replace("/", "_")

        S = 440  # per-panel size

        # --- Panel 1: RGB + 2D overlay ---
        overlay = rgb.copy()
        lm2d = _project(lm3d, K)
        for i, j in CONNECTIONS:
            cv2.line(overlay,
                     (int(lm2d[i, 0]), int(lm2d[i, 1])),
                     (int(lm2d[j, 0]), int(lm2d[j, 1])),
                     (0, 200, 220), 2)
        for pt in lm2d:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
        # Add label
        p1 = Image.fromarray(overlay).resize((S, S))
        d1 = ImageDraw.Draw(p1)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            f = ImageFont.load_default()
        d1.rectangle([0, 0, 100, 24], fill=(40, 40, 40))
        d1.text((6, 3), "RGB + 2D Joints", fill="white", font=f)

        # --- Panel 2: Depth map ---
        d_norm = np.clip(depth_uint16.astype(np.float32) / 1500.0, 0, 1)
        d_col = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        p2 = Image.fromarray(cv2.cvtColor(d_col, cv2.COLOR_BGR2RGB)).resize((S, S))
        d2 = ImageDraw.Draw(p2)
        d2.rectangle([0, 0, 70, 24], fill=(40, 40, 40))
        d2.text((6, 3), "Depth", fill="white", font=f)

        # --- Panel 3-5: 3D hand views ---
        p3 = _render_hand_3d(verts, lm3d, MANO_FACES, "front", S)
        p4 = _render_hand_3d(verts, lm3d, MANO_FACES, "side", S)
        p5 = _render_hand_3d(verts, lm3d, MANO_FACES, "angled", S)

        # --- Panel 6: Info ---
        p6 = Image.new("RGB", (S, S), (245, 245, 245))
        d6 = ImageDraw.Draw(p6)
        try:
            ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            ft = ImageFont.load_default()
            fn = ft
        y = 15
        d6.text((12, y), f"Sample: {stem}", fill="black", font=ft)
        y += 30
        for label, value in [
            ("Mesh vertices", f"{verts.shape[0]}"),
            ("Joints", f"{lm3d.shape[0]}"),
            ("Faces", f"{MANO_FACES.shape[0]}"),
            ("Wrist x", f"     {T_wrist[0,3]:+.3f} m"),
            ("Wrist y", f"     {T_wrist[1,3]:+.3f} m"),
            ("Wrist z", f"     {T_wrist[2,3]:+.3f} m"),
        ]:
            d6.text((12, y), f"{label}:", fill=(100, 100, 100), font=fn)
            d6.text((180, y), value, fill="black", font=fn)
            y += 22
        y += 10
        # Legend
        d6.text((12, y), "Legend:", fill="black", font=ft)
        y += 26
        d6.rectangle([12, y, 32, y + 16], fill=(240, 190, 165))
        d6.rectangle([12, y, 32, y + 16], outline=(180, 140, 120))
        d6.text((40, y), "Hand mesh (skin tone)", fill="black", font=fn)
        y += 22
        d6.line([12, y + 8, 32, y + 8], fill=(0, 180, 220), width=3)
        d6.text((40, y), "Skeleton (cyan)", fill="black", font=fn)
        y += 22
        d6.ellipse([14, y + 2, 30, y + 18], fill=(0, 255, 255), outline=(0, 100, 100))
        d6.text((40, y), "Joints (green dot)", fill="black", font=fn)
        y += 30
        d6.text((12, y), "Darker = farther from viewer", fill=(120, 120, 120), font=fn)

        # --- Assemble grid: 3 cols x 2 rows ---
        def _hstack(*imgs, gap=10, bg=(255, 255, 255)):
            w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
            h = max(i.height for i in imgs)
            out = Image.new("RGB", (w, h), bg)
            x = 0
            for i in imgs:
                out.paste(i, (x, (h - i.height) // 2))
                x += i.width + gap
            return out

        def _vstack(*imgs, gap=10, bg=(255, 255, 255)):
            w = max(i.width for i in imgs)
            h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
            out = Image.new("RGB", (w, h), bg)
            y = 0
            for i in imgs:
                out.paste(i, ((w - i.width) // 2, y))
                y += i.height + gap
            return out

        row1 = _hstack(p1, p2, p3)
        row2 = _hstack(p4, p5, p6)
        composite = _vstack(row1, row2)

        # --- Title bar ---
        final = Image.new("RGB", (composite.width, composite.height + 44), "white")
        final.paste(composite, (0, 44))
        d = ImageDraw.Draw(final)
        try:
            tf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            tf = ImageFont.load_default()
        d.text((15, 10), f"HUG Prediction: {stem}", fill="black", font=tf)

        out_path = output_dir / f"{stem}.png"
        final.save(out_path, quality=90)
        print(f"  Saved: {out_path}")

    print(f"\nDone! {len(pkl_files)} image(s) → {output_dir.resolve()}/")


if __name__ == "__main__":
    tyro.cli(main)

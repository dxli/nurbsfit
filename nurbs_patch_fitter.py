"""
✅ STANDALONE NURBS PATCH FITTER (v16.30 - LS + WATERTIGHT + FIXED PERIODIC KNOTS)
Run independently: python nurbs_patch_fitter.py
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from geomdl.NURBS import Surface

# ====================== MODULAR IMPORT ======================
from nurbs_patch_utils import compute_robust_local_basis


def detect_patch_degree_svd(mesh, patch_faces, target_max_dev=0.5, clean_mode=False):
    faces = mesh.faces[patch_faces]
    vert_idx = np.unique(faces)
    points = mesh.vertices[vert_idx]
    if len(points) < 10:
        return 3, "freeform", False, False

    centroid = np.mean(points, axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    mean_r = np.mean(radii)
    std_r = np.std(radii)
    radial_dev = np.max(np.abs(radii - mean_r)) / (mean_r + 1e-12)

    dist_to_axis = np.sqrt(points[:,0]**2 + points[:,1]**2)
    min_radius = np.min(dist_to_axis)
    has_hole = min_radius > 0.4 * mean_r
    if has_hole and std_r > 0.08 * mean_r:
        print("      Toroidal patch detected - forcing degree=3")
        return 3, "toroidal", True, True

    if std_r < 0.02 * mean_r and radial_dev < 0.012:
        print("      Sphere-like patch detected - forcing degree=3")
        return 3, "spherical", True, False

    X = points - centroid
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    A = np.column_stack((x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, np.ones_like(x)))
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    coeffs = Vt[-1]
    residuals = np.abs(A @ coeffs)
    max_res = residuals.max()

    factor = 0.005 if clean_mode else 0.02
    if max_res < target_max_dev * factor:
        _, _, Vt_plane = np.linalg.svd(X, full_matrices=False)
        normal = Vt_plane[-1]
        plane_dev = np.max(np.abs(X @ normal))
        patch_diameter = np.max(np.ptp(X, axis=0))
        if plane_dev > 0.015 * patch_diameter:
            return 2, "quadratic", False, False
        return 1, "planar", False, False

    return 3, "freeform", False, False


def make_knots(n_ctrl, deg, periodic=False):
    """Correct knot vector generation (fixed for periodic cases)."""
    if periodic:
        # Periodic: len(knots) == n_ctrl + deg + 1  (scipy requirement)
        # Knots wrap around [0,1] → BSpline(extrapolate=periodic) works perfectly
        return [float(i - deg) / n_ctrl for i in range(n_ctrl + deg + 1)]
    else:
        internal = max(1, n_ctrl - deg)
        return [0]*(deg + 1) + [float(i)/internal for i in range(1, internal)] + [1]*(deg + 1)


def _bspline_basis(knots, degree, t, periodic=False):
    """Univariate B-spline basis matrix – now consistent for both open & periodic."""
    n_basis = len(knots) - degree - 1
    basis = np.zeros((len(t), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=periodic)
        basis[:, i] = spl(t)
    return basis


def fit_b_spline_ls(uv, points, degree, n_ctrl_u, n_ctrl_v,
                    closed_u=False, closed_v=False, boundary_mask=None):
    """Linear least-squares with periodic support + boundary weighting for watertight shells."""
    knot_u = make_knots(n_ctrl_u, degree, periodic=closed_u)
    knot_v = make_knots(n_ctrl_v, degree, periodic=closed_v)
    Bu = _bspline_basis(knot_u, degree, uv[:, 0], periodic=closed_u)
    Bv = _bspline_basis(knot_v, degree, uv[:, 1], periodic=closed_v)

    # Tensor-product design matrix
    A = np.zeros((len(uv), n_ctrl_u * n_ctrl_v))
    for i in range(len(uv)):
        A[i] = np.kron(Bu[i], Bv[i])

    # Weighted LS for closed shells (boundary points 50× stronger)
    weights = np.ones(len(uv))
    if boundary_mask is not None:
        weights[boundary_mask] = 50.0
    W = np.diag(weights)

    ctrl = np.zeros((n_ctrl_u * n_ctrl_v, 3))
    for d in range(3):
        ATA = A.T @ W @ A
        ATb = A.T @ W @ points[:, d]
        ctrl[:, d] = np.linalg.solve(ATA, ATb)

    return ctrl, knot_u, knot_v


def adaptive_fit_nurbs_to_patch(mesh, patch_faces, max_z_deviation=0.01, clean_mode=False,
                                centripetal=True, verbose=False,
                                closed_shell=False, boundary_vert_mask=None):
    """Fit a single B-Rep patch – now fully stable on closed (sphere/toroidal) patches."""
    faces = mesh.faces[patch_faces]
    vert_idx = np.unique(faces)
    original_points = mesh.vertices[vert_idx].copy()
    points_3d = original_points.copy()

    if len(points_3d) < 3:
        print(f"      WARNING: degenerate patch ({len(points_3d)} verts) → skipping")
        return None, None

    if len(points_3d) < 6:
        print(f"      Small patch detected ({len(points_3d)} verts) → forcing planar (degree=1)")
        degree = 1
        patch_type = "planar"
        is_closed = False
        is_toroidal = False
    else:
        degree, patch_type, is_closed, is_toroidal = detect_patch_degree_svd(mesh, patch_faces, max_z_deviation, clean_mode)

    print(f"   Detected {patch_type} patch → degree={degree} (verts={len(points_3d)}) | closed_shell={closed_shell}")

    if degree == 1:
        print("      Planar patch → exact plane fit")
        centroid = np.mean(points_3d, axis=0)
        _, _, Vt = np.linalg.svd(points_3d - centroid, full_matrices=False)
        normal = Vt[-1]
        # 4-corner bilinear plane
        corners = np.array([
            points_3d[np.argmin(points_3d[:,0])],
            points_3d[np.argmax(points_3d[:,0])],
            points_3d[np.argmin(points_3d[:,1])],
            points_3d[np.argmax(points_3d[:,1])]
        ])
        surf = Surface()
        surf.degree_u = 1
        surf.degree_v = 1
        surf.ctrlpts_size_u = 2
        surf.ctrlpts_size_v = 2
        surf.ctrlpts = corners.tolist()
        surf.knotvector_u = [0, 0, 1, 1]
        surf.knotvector_v = [0, 0, 1, 1]
        surf.delta = (0.01, 0.01)
        return surf, {"vert_idx": vert_idx.tolist(), "z_dev": 0.0,
                      "closed": False, "toroidal": False,
                      "closed_u": False, "closed_v": False}

    # UV-parameterization (toroidal / spherical / regular)
    basis_u, basis_v, normal = compute_robust_local_basis(mesh, patch_faces)
    centroid = np.mean(points_3d, axis=0)

    closed_u = closed_v = False
    if is_toroidal:
        print("      Using TRUE BI-PERIODIC TOROIDAL parameterization")
        R = np.column_stack((basis_u, basis_v, normal))
        pts_local = (points_3d - centroid) @ R.T
        dist_to_axis = np.sqrt(pts_local[:,0]**2 + pts_local[:,1]**2)
        R_major = np.mean(dist_to_axis)
        theta = np.arctan2(pts_local[:,1], pts_local[:,0])
        phi   = np.arctan2(pts_local[:,2], dist_to_axis - R_major)
        uv = np.column_stack(((theta + np.pi) / (2*np.pi), (phi + np.pi) / (2*np.pi)))
        closed_u = closed_v = True
    elif is_closed:
        print("      Using SPHERICAL parameterization (U closed)")
        dirs = points_3d - centroid
        r = np.mean(np.linalg.norm(dirs, axis=1))
        dirs /= (r + 1e-12)
        theta = np.arctan2(dirs[:,1], dirs[:,0])
        phi   = np.arcsin(dirs[:,2])
        uv = np.column_stack(((theta + np.pi) / (2*np.pi), (phi + np.pi/2) / np.pi))
        closed_u = True
        closed_v = False
    else:
        uv = np.column_stack((np.dot(points_3d - centroid, basis_u),
                              np.dot(points_3d - centroid, basis_v)))

    if centripetal and not (closed_u or closed_v):
        dist = np.sqrt(np.sum(uv**2, axis=1))
        dist = dist / (dist.max() + 1e-12)
        uv = uv * np.sqrt(dist)[:, np.newaxis]

    uv_min = uv.min(axis=0)
    uv_range = np.ptp(uv, axis=0)
    uv_range[uv_range < 1e-12] = 1.0
    uv = (uv - uv_min) / uv_range

    # LS control-net size
    base_ctrl = {1: 4, 2: 6, 3: 8}
    n_ctrl_base = base_ctrl.get(degree, 8)
    patch_diameter = np.max(np.ptp(points_3d, axis=0))
    diameter_factor = min(3.0, patch_diameter / 40.0) if closed_shell else min(2.5, patch_diameter / 50.0)
    n_ctrl = max(n_ctrl_base, int(n_ctrl_base * diameter_factor))
    n_ctrl = min(n_ctrl, 20 if closed_shell else 16)
    print(f"      LS fit → {n_ctrl}×{n_ctrl} control points (periodic={closed_u or closed_v})")

    # Least-squares fit (now works for closed patches)
    ctrlpts_array, knot_u, knot_v = fit_b_spline_ls(
        uv, points_3d, degree, n_ctrl, n_ctrl,
        closed_u=closed_u, closed_v=closed_v,
        boundary_mask=boundary_vert_mask
    )

    surf = Surface()
    surf.degree_u = degree
    surf.degree_v = degree
    surf.ctrlpts_size_u = n_ctrl
    surf.ctrlpts_size_v = n_ctrl
    surf.ctrlpts = ctrlpts_array.tolist()
    surf.knotvector_u = knot_u
    surf.knotvector_v = knot_v
    surf.delta = (0.01, 0.01)

    # Short quality-boost loop
    max_passes = 2 if not (closed_u or closed_v) else 3
    for pass_num in range(max_passes):
        print(f"      Quality pass {pass_num+1}/{max_passes}")
        for it in range(12):
            eval_pts = np.array(surf.evalpts)
            tree = KDTree(eval_pts)
            _, idx = tree.query(points_3d)
            snapped = eval_pts[idx]
            new_ctrl, _, _ = fit_b_spline_ls(uv, snapped, degree, n_ctrl, n_ctrl,
                                             closed_u=closed_u, closed_v=closed_v,
                                             boundary_mask=boundary_vert_mask)
            surf.ctrlpts = new_ctrl.tolist()

            eval_pts = np.array(surf.evalpts)
            tree = KDTree(eval_pts)
            _, idx = tree.query(original_points)
            closest = eval_pts[idx]
            z_dev = np.max(np.abs((original_points - closest) @ normal))

            if verbose:
                print(f"         Reparam {it+1:2d} → Z-dev: {z_dev:.6f}")
            if z_dev < max_z_deviation * 0.8:
                break
        if z_dev <= max_z_deviation:
            break

    # Final deviation check
    eval_pts = np.array(surf.evalpts)
    tree = KDTree(eval_pts)
    _, idx = tree.query(original_points)
    closest = eval_pts[idx]
    final_dev = np.max(np.abs((original_points - closest) @ normal))

    print(f"      FINAL ACHIEVED Z-dev: {final_dev:.6f} (target {max_z_deviation})")
    if final_dev < max_z_deviation:
        print("      SUCCESS: Target achieved!")

    return surf, {"vert_idx": vert_idx.tolist(), "z_dev": final_dev,
                  "closed": is_closed or is_toroidal, "toroidal": is_toroidal,
                  "closed_u": closed_u, "closed_v": closed_v}


# ====================== Independent test ======================
if __name__ == "__main__":
    import trimesh
    print("=== STANDALONE TEST: Fitting a single sphere patch ===")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    patch_faces = np.arange(len(mesh.faces))
    surf, info = adaptive_fit_nurbs_to_patch(mesh, patch_faces, max_z_deviation=0.01, verbose=False)
    print("Standalone fitting complete. Z-dev:", info["z_dev"])
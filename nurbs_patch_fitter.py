"""
✅ STANDALONE NURBS PATCH FITTER (v16.28)
Run independently: python nurbs_patch_fitter.py
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from geomdl.NURBS import Surface


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


def compute_robust_local_basis(mesh, patch_faces):
    faces = mesh.faces[patch_faces]
    vert_idx = np.unique(faces)
    points = mesh.vertices[vert_idx]
    centroid = np.mean(points, axis=0)
    X = points - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        normal = np.array([0., 0., 1.])
    else:
        normal /= norm
    arb = np.array([0., 0., 1.]) if abs(normal[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(normal, arb)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    return u, v, normal


def adaptive_fit_nurbs_to_patch(mesh, patch_faces, max_z_deviation=0.01, clean_mode=False, centripetal=True, verbose=False):
    """Fit a single B-Rep patch to a NURBS surface (standalone version)."""
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

    print(f"   Detected {patch_type} patch → degree={degree} (verts={len(points_3d)})")

    if degree == 1:
        print("      Planar patch → exact plane fit")
        centroid = np.mean(points_3d, axis=0)
        _, _, Vt = np.linalg.svd(points_3d - centroid, full_matrices=False)
        normal = Vt[-1]
        corners = np.array([points_3d[np.argmin(points_3d[:,0])],
                            points_3d[np.argmax(points_3d[:,0])],
                            points_3d[np.argmin(points_3d[:,1])],
                            points_3d[np.argmax(points_3d[:,1])]])
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

    # UV-parameterization + adaptive grid + quality boost (same as v16.28)
    basis_u, basis_v, normal = compute_robust_local_basis(mesh, patch_faces)
    centroid = np.mean(points_3d, axis=0)

    closed_u = False
    closed_v = False
    if is_toroidal:
        print("      Using TRUE BI-PERIODIC TOROIDAL parameterization (U+V closed)")
        R = np.column_stack((basis_u, basis_v, normal))
        pts_local = (points_3d - centroid) @ R.T
        dist_to_axis = np.sqrt(pts_local[:,0]**2 + pts_local[:,1]**2)
        R_major = np.mean(dist_to_axis)
        theta = np.arctan2(pts_local[:,1], pts_local[:,0])
        phi   = np.arctan2(pts_local[:,2], dist_to_axis - R_major)
        uv = np.column_stack(((theta + np.pi) / (2*np.pi),
                              (phi   + np.pi) / (2*np.pi)))
        closed_u = True
        closed_v = True
    elif is_closed:
        print("      Using SPHERICAL parameterization (U closed, V open with poles)")
        dirs = points_3d - centroid
        r = np.mean(np.linalg.norm(dirs, axis=1))
        dirs /= (r + 1e-12)
        dirs = np.clip(dirs, -1.0, 1.0)
        theta = np.arctan2(dirs[:,1], dirs[:,0])
        phi   = np.arcsin(dirs[:,2])
        uv = np.column_stack(((theta + np.pi) / (2*np.pi),
                              (phi + np.pi/2) / np.pi))
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

    # Adaptive grid (96×96 min for spheres)
    base_size = {1: 8, 2: 16, 3: 32}[degree]
    if closed_u or closed_v:
        base_size = max(base_size, 96)
    patch_diameter = np.max(np.ptp(points_3d, axis=0))
    diameter_factor = min(3.0, patch_diameter / 40.0)
    grid_size = int(base_size * diameter_factor)
    grid_size = max(32, min(128, grid_size))

    print(f"      Adaptive grid → {grid_size}×{grid_size} (degree={degree}, closed_u={closed_u}, closed_v={closed_v})")

    grid_uv = np.mgrid[0:1:complex(0, grid_size), 0:1:complex(0, grid_size)].reshape(2, -1).T
    grid_3d = griddata(uv, points_3d, grid_uv, method='cubic')
    grid_3d = np.nan_to_num(grid_3d, nan=0.0)

    surf = Surface()
    surf.degree_u = degree
    surf.degree_v = degree

    if closed_u and closed_v:
        g = grid_3d.reshape(grid_size, grid_size, 3)
        grid_closed = np.zeros((grid_size + 1, grid_size + 1, 3))
        grid_closed[:-1, :-1] = g
        grid_closed[:-1, -1]  = g[:, 0]
        grid_closed[-1, :-1]  = g[0, :]
        grid_closed[-1, -1]   = g[0, 0]
        surf.ctrlpts_size_u = grid_size + 1
        surf.ctrlpts_size_v = grid_size + 1
        surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
    elif closed_u:
        g = grid_3d.reshape(grid_size, grid_size, 3)
        south_pole = g[0, 0]
        north_pole = g[-1, 0]
        g[0, :] = south_pole
        g[-1, :] = north_pole
        grid_closed = np.zeros((grid_size, grid_size + 1, 3))
        grid_closed[:, :grid_size] = g
        grid_closed[:, grid_size]  = g[:, 0]
        surf.ctrlpts_size_u = grid_size + 1
        surf.ctrlpts_size_v = grid_size
        surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
    else:
        surf.ctrlpts_size_u = grid_size
        surf.ctrlpts_size_v = grid_size
        surf.ctrlpts = grid_3d.tolist()

    def make_knots(n_ctrl, deg):
        internal = max(1, n_ctrl - deg)
        return [0]*(deg + 1) + [float(i)/internal for i in range(1, internal)] + [1]*(deg + 1)

    surf.knotvector_u = make_knots(surf.ctrlpts_size_u, degree)
    surf.knotvector_v = make_knots(surf.ctrlpts_size_v, degree)
    surf.delta = (0.01, 0.01)

    # Quality-boost loop
    max_passes = 5
    for pass_num in range(max_passes):
        print(f"      Quality pass {pass_num+1}/{max_passes} (grid={grid_size})")
        for it in range(25):
            eval_pts = np.array(surf.evalpts)
            tree = KDTree(eval_pts)
            _, idx = tree.query(points_3d)
            snapped = eval_pts[idx]
            new_grid_3d = griddata(uv, snapped, grid_uv, method='cubic')
            new_grid_3d = np.nan_to_num(new_grid_3d, nan=0.0)

            if closed_u and closed_v:
                g = new_grid_3d.reshape(grid_size, grid_size, 3)
                grid_closed = np.zeros((grid_size + 1, grid_size + 1, 3))
                grid_closed[:-1, :-1] = g
                grid_closed[:-1, -1]  = g[:, 0]
                grid_closed[-1, :-1]  = g[0, :]
                grid_closed[-1, -1]   = g[0, 0]
                surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
            elif closed_u:
                g = new_grid_3d.reshape(grid_size, grid_size, 3)
                south_pole = g[0, 0]
                north_pole = g[-1, 0]
                g[0, :] = south_pole
                g[-1, :] = north_pole
                grid_closed = np.zeros((grid_size, grid_size + 1, 3))
                grid_closed[:, :grid_size] = g
                grid_closed[:, grid_size]  = g[:, 0]
                surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
            else:
                surf.ctrlpts = new_grid_3d.tolist()

            eval_pts = np.array(surf.evalpts)
            tree = KDTree(eval_pts)
            _, idx = tree.query(original_points)
            closest = eval_pts[idx]
            z_dev = np.max(np.abs((original_points - closest) @ normal))

            if verbose:
                print(f"         Reparam {it+1:2d} → Z-dev: {z_dev:.6f}")

            if z_dev < max_z_deviation:
                break

        if z_dev <= max_z_deviation:
            break
        if pass_num < max_passes - 1:
            grid_size = min(128, int(grid_size * 2.0))
            print(f"      Z-dev still {z_dev:.6f} → increasing grid to {grid_size} and re-fitting")
            grid_uv = np.mgrid[0:1:complex(0, grid_size), 0:1:complex(0, grid_size)].reshape(2, -1).T
            grid_3d = griddata(uv, points_3d, grid_uv, method='cubic')
            grid_3d = np.nan_to_num(grid_3d, nan=0.0)
            surf = Surface()
            surf.degree_u = degree
            surf.degree_v = degree
            if closed_u and closed_v:
                g = grid_3d.reshape(grid_size, grid_size, 3)
                grid_closed = np.zeros((grid_size + 1, grid_size + 1, 3))
                grid_closed[:-1, :-1] = g
                grid_closed[:-1, -1]  = g[:, 0]
                grid_closed[-1, :-1]  = g[0, :]
                grid_closed[-1, -1]   = g[0, 0]
                surf.ctrlpts_size_u = grid_size + 1
                surf.ctrlpts_size_v = grid_size + 1
                surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
            elif closed_u:
                g = grid_3d.reshape(grid_size, grid_size, 3)
                south_pole = g[0, 0]
                north_pole = g[-1, 0]
                g[0, :] = south_pole
                g[-1, :] = north_pole
                grid_closed = np.zeros((grid_size, grid_size + 1, 3))
                grid_closed[:, :grid_size] = g
                grid_closed[:, grid_size]  = g[:, 0]
                surf.ctrlpts_size_u = grid_size + 1
                surf.ctrlpts_size_v = grid_size
                surf.ctrlpts = grid_closed.reshape(-1, 3).tolist()
            else:
                surf.ctrlpts_size_u = grid_size
                surf.ctrlpts_size_v = grid_size
                surf.ctrlpts = grid_3d.tolist()
            surf.knotvector_u = make_knots(surf.ctrlpts_size_u, degree)
            surf.knotvector_v = make_knots(surf.ctrlpts_size_v, degree)
            surf.delta = (0.01, 0.01)

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


# ====================== Independent test (run this file directly) ======================
if __name__ == "__main__":
    import trimesh
    print("=== STANDALONE TEST: Fitting a single sphere patch ===")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    patch_faces = np.arange(len(mesh.faces))  # whole mesh as one patch
    surf, info = adaptive_fit_nurbs_to_patch(mesh, patch_faces, max_z_deviation=0.01, verbose=True)
    print("Standalone fitting complete. Z-dev:", info["z_dev"])
"""
✅ NURBS PATCH UTILITIES (v16.28)
Standalone helpers for local basis, UV parameterization, etc.
Run independently if needed for debugging.
"""

import numpy as np


def compute_robust_local_basis(mesh, patch_faces):
    """Compute robust orthonormal basis (u, v, normal) for a patch using SVD.
    
    This is the exact same logic as before – now isolated for modularity.
    Used by adaptive_fit_nurbs_to_patch() for UV parameterization.
    """
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

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.linalg import eig

PATCH_A_FILE = 'PATH_TEST_FILE_A'
PATCH_B_FILE = 'PATH_TEST_FILE_B'
PATCH_C_FILE = 'PATH_TEST_FILE_C'


def visualize_implicit_quadric(coefficients, points, normalized_distance, grid_size=50):
    """
    Visualize the implicit quadric surface using its coefficients.

    Parameters:
    - coefficients (numpy.ndarray): Quadric coefficients [A, B, C, D, E, F, G, H, I, J].
    - points (numpy.ndarray): Original 3D points.
    - normalized_distance (float): Normalized least squares algebraic distance.
    - grid_size (int): Resolution of the 3D grid for visualization.
    """
    # Extract coefficients
    A, B, C, D, E, F, G, H, I, J = coefficients

    # Create a 3D grid of points
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = np.linspace(z_min, z_max, grid_size)

    X, Y, Z = np.meshgrid(x, y, z)

    # Evaluate the implicit quadric equation: f(x, y, z) = 0
    F = (A * X ** 2 + B * Y ** 2 + C * Z ** 2 +
         D * X * Y + E * X * Z + F * Y * Z +
         G * X + H * Y + I * Z + J)

    # Use the Marching Cubes algorithm to extract the surface where F = 0
    verts, faces, normals, values = measure.marching_cubes(F, level=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(verts[:, 0] * (x_max - x_min) / grid_size + x_min,
                    verts[:, 1] * (y_max - y_min) / grid_size + y_min,
                    verts[:, 2] * (z_max - z_min) / grid_size + z_min,
                    triangles=faces, cmap='viridis', alpha=0.6)

    # Plot the original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10, label='Original Points')

    # Display the normalized distance in scientific notation
    normalized_distance_formatted = f"{normalized_distance:.3e}"
    ax.text2D(0.05, 0.95, f"Fitting Error: {normalized_distance_formatted}",
              transform=ax.transAxes, fontsize=12, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Implicit Quadric Surface')
    ax.legend()
    plt.axis('equal')
    plt.show()


def taubin_fit_with_gradient(points):
    # Validate input points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points should be an N x 3 array.")

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Design matrix for the quadric surface Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    D = np.column_stack((
        X ** 2, Y ** 2, Z ** 2,  # Ax^2, By^2, Cz^2
        X * Y, X * Z, Y * Z,  # Dxy, Exz, Fyz
        X, Y, Z,  # Gx, Hy, Iz
        np.ones_like(X)  # Constant J term
    ))

    # Partial derivatives with respect to x, y, z (N x 10 matrices)
    G = np.array([D,D,D])
    G[0] = np.column_stack((
        2 * X, np.zeros_like(X), np.zeros_like(X),
        Y, Z, np.zeros_like(X),
        np.ones_like(X), np.zeros_like(X), np.zeros_like(X),
        np.zeros_like(X)
    ))

    G[1] = np.column_stack((
        np.zeros_like(Y), 2 * Y, np.zeros_like(Y),
        X, np.zeros_like(Y), Z,
        np.zeros_like(Y), np.ones_like(Y), np.zeros_like(Y),
        np.zeros_like(Y)
    ))

    G[2] = np.column_stack((
        np.zeros_like(Z), np.zeros_like(Z), 2 * Z,
        np.zeros_like(Z), X, Y,
        np.zeros_like(Z), np.zeros_like(Z), np.ones_like(Z),
        np.zeros_like(Z)
    ))

    # Covariance matrix (M = D^T D)
    M = D.T @ D

    # Constraint matrix (N) from the gradient terms
    N = np.zeros((10, 10))  # Initialize N as a 10x10 matrix
    for i in range(3):
        N += G[i].T @ G[i]  # Accumulate gradient contributions
    #N /= len(points)  # Normalize by the number of points

    # Solve the generalized eigenvalue problem: (M - λN)c = 0
    eigenvalues, eigenvectors = eig(M, N)

    # Find the eigenvector corresponding to the smallest eigenvalue
    min_idx = np.argmin(eigenvalues.real)
    coefficients = eigenvectors[:, min_idx].real

    # Normalize the coefficients to have unit norm
    norm_coeff = np.linalg.norm(coefficients)
    if norm_coeff < 1e-8:
        raise ValueError("Degenerate coefficients result in numerical issues.")
    coefficients /= norm_coeff

    # Compute the error: c^T M c / c^T N c
    cT_M_c = coefficients.T @ M @ coefficients
    cT_N_c = coefficients.T @ N @ coefficients
    error = cT_M_c / cT_N_c

    return coefficients, error


def compute_matrices(points):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Design matrix
    D = np.column_stack((
        X ** 2, Y ** 2, Z ** 2,  # Ax^2, By^2, Cz^2
        X * Y, X * Z, Y * Z,  # Dxy, Exz, Fyz
        X, Y, Z,  # Gx, Hy, Iz
        np.ones_like(X)  # Constant J term
    ))

    # Covariance matrix (M = D^T D)
    M = D.T @ D

    # Gradient terms for N
    G = np.array([D,D,D])
    G[0] = np.column_stack((
        2 * X, np.zeros_like(X), np.zeros_like(X),
        Y, Z, np.zeros_like(X),
        np.ones_like(X), np.zeros_like(X), np.zeros_like(X),
        np.zeros_like(X)
    ))

    G[1] = np.column_stack((
        np.zeros_like(Y), 2 * Y, np.zeros_like(Y),
        X, np.zeros_like(Y), Z,
        np.zeros_like(Y), np.ones_like(Y), np.zeros_like(Y),
        np.zeros_like(Y)
    ))

    G[2] = np.column_stack((
        np.zeros_like(Z), np.zeros_like(Z), 2 * Z,
        np.zeros_like(Z), X, Y,
        np.zeros_like(Z), np.zeros_like(Z), np.ones_like(Z),
        np.zeros_like(Z)
    ))

    # Constraint matrix (N)
    N = np.zeros((10, 10))  # Initialize N as a 10x10 matrix
    for i in range(3):
        N += G[i].T @ G[i]  # Accumulate gradient contributions
    # N /= len(points)  # Normalize by the number of points

    return M, N


def test_merge_matrices(points1, points2):
    # Compute M, N for each set of points
    M1, N1 = compute_matrices(points1)
    M2, N2 = compute_matrices(points2)

    # Merge M and N
    M_merged = M1 + M2
    N_merged = N1 + N2
    # N_merged = (N1 * len(points1) + N2 * len(points2)) / (len(points1) + len(points2))

    # Merge the points
    points_merged = np.vstack([points1, points2])

    # Compute M, N directly from merged points
    M_direct, N_direct = compute_matrices(points_merged)

    # Compare the results
    print("Difference in M:", np.linalg.norm(M_merged - M_direct))
    print("Difference in N:", np.linalg.norm(N_merged - N_direct))

    coefficients1, error1 = solve_taubin_from_matrices(M_merged, N_merged)
    coefficients2, error2 = solve_taubin_from_matrices(M_direct, N_direct)
    print("Merged fitting error:", error1)
    print("Direct fitting error:", error2)

    visualize_implicit_quadric(coefficients1, points_merged, error1)
    visualize_implicit_quadric(coefficients2, points_merged, error2)


def solve_taubin_from_matrices(M, N):
    """
    Solves for the quadric coefficients and fitting error given covariance matrix M and constraint matrix N.

    Parameters:
    - M (ndarray): Covariance matrix of size 10x10 (M = D^T D).
    - N (ndarray): Constraint matrix of size 10x10 (N is derived from gradients).

    Returns:
    - coefficients (ndarray): Normalized quadric coefficients (10x1 vector).
    - error (float): Fitting error computed as c^T M c / c^T N c.
    """
    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eig(M, N)

    # Find the eigenvector corresponding to the smallest eigenvalue
    min_idx = np.argmin(eigenvalues.real)
    coefficients = eigenvectors[:, min_idx].real  # Select the eigenvector

    # Normalize the coefficients to have unit norm
    norm_coeff = np.linalg.norm(coefficients)
    if norm_coeff < 1e-8:
        raise ValueError("Degenerate coefficients result in numerical issues.")
    coefficients /= norm_coeff

    # Compute the fitting error
    cT_M_c = coefficients.T @ M @ coefficients
    cT_N_c = coefficients.T @ N @ coefficients
    error = cT_M_c / cT_N_c

    return coefficients, error


if __name__ == '__main__':


    patch_a_file = PATCH_A_FILE
    patch_b_file = PATCH_B_FILE
    patch_c_file = PATCH_C_FILE

    pointcloud = trimesh.load(patch_a_file)
    points1 = np.asarray(pointcloud.vertices)
    pointcloud = trimesh.load(patch_c_file)
    points2 = np.asarray(pointcloud.vertices)


    test_merge_matrices(points1, points2)

    print('__playing with quadrics__')
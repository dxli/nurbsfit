import numpy as np
import torch
import trimesh

from geomdl import NURBS
from geomdl import exchange
from geomdl import knotvector

import matplotlib.pyplot as plt
from trimesh.transformations import rotation_matrix
from trim_meshes import trim_mesh, compute_distances_to_mesh, trim_mesh_by_distance

from utils import *

EXPERIMENTS_PATH = 'DATA_PATH'
GOCOPP_BASE_PATH = 'GOCOPP_DATA_PATH'

def mesh_from_nurbs(filename, ctrlpts_size_u, ctrlpts_size_v, ctrlpts, sample_size=100, write_off=True):
    # Create a NURBS surface instance
    surf = NURBS.Surface()
    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = ctrlpts_size_u
    surf.ctrlpts_size_v = ctrlpts_size_v

    surf.ctrlpts = ctrlpts

    # Set knot vectors
    surf.knotvector_u = knotvector.generate(surf.degree[0], surf.ctrlpts_size_u)
    surf.knotvector_v = knotvector.generate(surf.degree[1], surf.ctrlpts_size_v)

    surf.sample_size = sample_size
    surf.render()

    surfpts = np.array(surf.evalpts)

    if write_off:
        exchange.export_off(surf, filename + "_mesh.off")
    return surfpts

def mapping_from_nurbs(ctrlpts_size_u, ctrlpts_size_v, ctrlpts, uv_points, knots_u=None, knots_v=None, sample_size=100):
    surf = NURBS.Surface()
    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = ctrlpts_size_u
    surf.ctrlpts_size_v = ctrlpts_size_v

    surf.ctrlpts = ctrlpts
    # If knots are provided, use them; otherwise, generate them
    if knots_u is not None and knots_v is not None:
        surf.knotvector_u = knots_u
        surf.knotvector_v = knots_v
    else:
        # Set knot vectors
        surf.knotvector_u = knotvector.generate(surf.degree[0], surf.ctrlpts_size_u)
        surf.knotvector_v = knotvector.generate(surf.degree[1], surf.ctrlpts_size_v)

    surf.sample_size = sample_size

    points = []
    #eval the uv points
    for uv_point in uv_points:
        # clamp uv points between 0 and 1
        uv_point = (min(max(uv_point[0], 0), 1), min(max(uv_point[1], 0), 1))

        point = surf.evaluate_single((uv_point[0], uv_point[1]))
        points.append(point)

    # surfpts = np.array(surf.evalpts)
    points = np.array(points)
    return points

def points_from_nurbs( ctrlpts_size_u, ctrlpts_size_v, ctrlpts, knots_u= None, knots_v=None, sample_size=100):
    # Create a NURBS surface instance
    surf = NURBS.Surface()
    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = ctrlpts_size_u
    surf.ctrlpts_size_v = ctrlpts_size_v

    surf.ctrlpts = ctrlpts

    # Set knot vectors
    if knots_u is not None and knots_v is not None:
        surf.knotvector_u = knots_u
        surf.knotvector_v = knots_v
    else:
        surf.knotvector_u = knotvector.generate(surf.degree[0], surf.ctrlpts_size_u)
        surf.knotvector_v = knotvector.generate(surf.degree[1], surf.ctrlpts_size_v)

    surf.sample_size = sample_size
    # surf.render()
    #
    surfpts = np.array(surf.evalpts)

    return surfpts



def find_minimal_bounding_box_3d(projected_points, plane_c):
    """
    Find the minimal bounding box that encloses a set of 3D points projected onto a plane.
    """
    from scipy.spatial import ConvexHull
    # Project all points onto the plane
    # projected_points = np.array([projected_points(point, a, b, c, d) for point in points])

    # Extract the x and y coordinates of the projected points
    projected_points_2d = projected_points[:, :2]

    # Compute the convex hull of the projected points
    hull = ConvexHull(projected_points_2d)
    hull_points = projected_points_2d[hull.vertices]

    # Find the minimal bounding box using rotating calipers
    min_area = np.inf
    best_box = None

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        edge_direction = edge / edge_length
        perpendicular_direction = np.array([-edge_direction[1], edge_direction[0]])

        # Project all hull points onto the edge and perpendicular directions
        projections_on_edge = np.dot(hull_points, edge_direction)
        projections_on_perpendicular = np.dot(hull_points, perpendicular_direction)

        min_proj_edge = np.min(projections_on_edge)
        max_proj_edge = np.max(projections_on_edge)
        min_proj_perpendicular = np.min(projections_on_perpendicular)
        max_proj_perpendicular = np.max(projections_on_perpendicular)

        width = max_proj_edge - min_proj_edge
        height = max_proj_perpendicular - min_proj_perpendicular
        area = width * height

        if area < min_area:
            min_area = area
            best_box = (min_proj_edge, max_proj_edge, min_proj_perpendicular, max_proj_perpendicular, edge_direction,
                        perpendicular_direction)

    # Reconstruct the coordinates of the minimal bounding box corners
    min_proj_edge, max_proj_edge, min_proj_perpendicular, max_proj_perpendicular, edge_direction, perpendicular_direction = best_box
    corner1 = min_proj_edge * edge_direction + min_proj_perpendicular * perpendicular_direction
    corner2 = max_proj_edge * edge_direction + min_proj_perpendicular * perpendicular_direction
    corner3 = max_proj_edge * edge_direction + max_proj_perpendicular * perpendicular_direction
    corner4 = min_proj_edge * edge_direction + max_proj_perpendicular * perpendicular_direction

    # Convert the 2D corners back to 3D points on the plane
    corners_3d = np.array([
        corner1,
        corner2,
        corner3,
        corner4,
    ])

    zz_grid = (-plane_c[0] * corners_3d[:, 0] - plane_c[1] * corners_3d[:, 1] - plane_c[3]) / plane_c[2]
    corners = np.stack([ corners_3d[:, 0],  corners_3d[:, 1], zz_grid], axis=-1)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='r', marker='o')
    # ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c='b', marker='o')
    # plt.show()

    return corners

def create_grid_from_plane(plane_c, points, npoints_x, npoints_y):
    plane = PlaneProjection(plane_c[0], plane_c[1], plane_c[2], plane_c[3])
    projected_points = plane.project_points(points)

    # plot_planes_points(plane_c, projected_points)

    corners_3d = find_minimal_bounding_box_3d(projected_points, plane_c)

    [P1, P2, P3, P4] = corners_3d
    [a, b, c, d] = plane_c
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    P4 = np.array(P4)

    # Create the parameter grid
    u = np.linspace(0, 1, npoints_x)
    v = np.linspace(0, 1, npoints_y)
    U, V = np.meshgrid(u, v)

    # Compute the grid points in the plane
    grid_points = np.zeros((npoints_x, npoints_y, 3))
    for i in range(npoints_x):
        for j in range(npoints_y):
            uv_point = (1 - U[i, j]) * (1 - V[i, j]) * P1 + U[i, j] * (1 - V[i, j]) * P2 + U[i, j] * V[i, j] * P3 + (
                        1 - U[i, j]) * V[i, j] * P4
            # Ensure the point lies in the plane (this is theoretically true if the corners are correct)
            if abs(a * uv_point[0] + b * uv_point[1] + c * uv_point[2] + d) < 1e-6:
                grid_points[i, j] = uv_point

    # grid_points = np.array(grid_points)

    # visualize_projected_grid_points(pca_points, grid_points, projected_points)
    return grid_points

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2.
    :param vec1: A 3d "source" vector.
    :param vec2: A 3d "destination" vector.
    :return mat: A transformation matrix (3x3) which, when applied to vec1, aligns it with vec2.
    """
    a = normalize(vec1)
    b = normalize(vec2)

    # Cross product of the two vectors
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        # The vectors are parallel; return the identity matrix
        return np.eye(3)
    elif np.isclose(c, -1.0):
        # The vectors are antiparallel; find an axis to rotate around
        # We can choose any axis orthogonal to vec1 for a 180 degree rotation
        orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(a, orthogonal_axis)
        v = normalize(v)  # Make sure this axis is normalized
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * kmat.dot(kmat)  # 180 degree rotation

    # Normal case
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def get_rotation_matrix( plane_c):
    normal_vector = np.array([plane_c[0], plane_c[1], plane_c[2]])
    z_axis = np.array([0, 1, 1])/np.linalg.norm([0, 1, 1])
    # Compute the rotation matrix to align the normal vector with the z-axis
    rotation_matrix = rotation_matrix_from_vectors(normal_vector, z_axis)

    return rotation_matrix
def generate_triangular_mesh(grid_points):
    """
    Generate a triangular mesh from a grid of control points.

    Parameters:
    - grid_points: A numpy array of shape (nx, ny, 3) containing points (x, y, z) in the grid.

    Returns:
    - vertices: List of vertices in the mesh.
    - triangles: List of triangles, where each triangle is defined by indices of its vertices.
    """
    nx, ny, _ = grid_points.shape
    vertices = grid_points.reshape(-1, 3)
    triangles = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            # Get the indices of the four corner points of the quad
            p1 = i * ny + j
            p2 = i * ny + (j + 1)
            p3 = (i + 1) * ny + j
            p4 = (i + 1) * ny + (j + 1)

            # Define the two triangles in the quad
            triangles.append([p1, p2, p3])
            triangles.append([p3, p2, p4])

    return vertices, triangles


def scale_points_to_unit_cube_back(points):
    """Scale points to fit within the unit cube [0, 1] x [0, 1] x [0, 1]."""
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Check for identical min and max to avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero by setting range to 1 for constant values

    scaled_points = (points - min_vals) / range_vals
    return scaled_points, min_vals, max_vals


def scale_points_to_unit_cube(points):
    """Scale points to fit within the unit cube [0, 1] x [0, 1] x [0, 1],
    with the z-axis scaled based on the mean of x and y scaling, and centered."""

    # Get the min and max values for each axis (x, y, z)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Compute the scaling range for x and y axes
    range_vals_xy = max_vals[:2] - min_vals[:2]
    range_vals_xy[range_vals_xy == 0] = 1  # Avoid division by zero for constant x or y values

    # Scale x and y axes to the [0, 1] range
    scaled_points = np.zeros_like(points)
    scaled_points[:, :2] = (points[:, :2] - min_vals[:2]) / range_vals_xy

    # Compute the z scaling based on the mean of x and y scaling
    z_scale = np.mean(range_vals_xy)

    # Center the z-axis and scale based on z_scale
    # z_center = (max_vals[2] + min_vals[2]) / 2
    scaled_points[:, 2] = (points[:, 2] - min_vals[2]) / z_scale

    return scaled_points, min_vals, max_vals

def scale_points_same_range_back(points, min_vals, max_vals):
    """Scale points to fit within the unit cube [0, 1] x [0, 1] x [0, 1]."""
    # Check for identical min and max to avoid division by zero

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero by setting range to 1 for constant values

    scaled_points = (points - min_vals) / range_vals
    return scaled_points

def scale_points_same_range(points, min_vals, max_vals):
    """Scale points to fit within the unit cube [0, 1] x [0, 1] x [0, 1]."""
    # Check for identical min and max to avoid division by zero
    # Compute the scaling range for x and y axes
    range_vals_xy = max_vals[:2] - min_vals[:2]
    range_vals_xy[range_vals_xy == 0] = 1  # Avoid division by zero for constant x or y values

    # Scale x and y axes to the [0, 1] range
    scaled_points = np.zeros_like(points)
    scaled_points[:, :2] = (points[:, :2] - min_vals[:2]) / range_vals_xy

    # Compute the z scaling based on the mean of x and y scaling
    z_scale = np.mean(range_vals_xy)

    # Center the z-axis and scale based on z_scale
    # z_center = (max_vals[2] + min_vals[2]) / 2
    scaled_points[:, 2] = (points[:, 2] - min_vals[2]) / z_scale

    return scaled_points


def unscale_points_from_unit_cube_back(points, min_vals, max_vals):
    """Undo the scaling of points from the unit cube back to the original range."""
    return points * (max_vals - min_vals) + min_vals


def unscale_points_from_unit_cube(scaled_points, min_vals, max_vals):
    """
    Reverse the scaling of points from the unit cube [0, 1] x [0, 1] x [0, 1]
    to the original coordinate space.

    Args:
        scaled_points (np.ndarray): Points scaled within the unit cube.
        min_vals (np.ndarray): Original minimum values of the points.
        max_vals (np.ndarray): Original maximum values of the points.

    Returns:
        np.ndarray: Original points in their original coordinate space.
    """
    # Compute the original range for x and y axes
    range_vals_xy = max_vals[:2] - min_vals[:2]
    range_vals_xy[range_vals_xy == 0] = 1  # Handle constant values in x or y

    # Compute the z scaling based on the mean of x and y scaling
    z_scale = np.mean(range_vals_xy)

    # Reconstruct original points
    original_points = np.zeros_like(scaled_points)

    # Reverse x and y scaling
    original_points[:, :2] = scaled_points[:, :2] * range_vals_xy + min_vals[:2]

    # Reverse z scaling (centered and scaled)
    # z_center = (max_vals[2] + min_vals[2]) / 2
    original_points[:, 2] = scaled_points[:, 2] * z_scale +  min_vals[2]

    return original_points

def transform_points_to_local(primitive_points):

    # # #scale the points to unit cub
    primitive_points, min_vals, max_vals = scale_points_to_unit_cube(primitive_points)


    return  primitive_points, min_vals, max_vals


def transform_points_to_global(primitive_points, ctrl_points, min_vals, max_vals):
    ctrl_points_cpu = ctrl_points.cpu().detach().numpy()
    # #rotate back the points and ctrl points

    # # #unscale the points
    uv_dims = ctrl_points_cpu.shape
    ctrl_points_cpu_list = unscale_points_from_unit_cube(ctrl_points_cpu.reshape(-1,3), min_vals, max_vals)
    ctrl_points_cpu = ctrl_points_cpu_list.reshape(uv_dims)


    return primitive_points, ctrl_points_cpu

def save_nurbs_patch(path_save, shape_name, ctrl_points, primitive_points, i, epsilon=0.1, save_trimmed=True, save_points=True, save_control_polygon=True, save_colored_mesh=True):
    mesh = trimesh.load(path_save + 'surface/' + shape_name + '_' + str(i) + '_mesh.off')
    distances = compute_distances_to_mesh(primitive_points, mesh)
    colors = map_distances_to_colors(distances)
    random_color = get_random_color_from_colormap('viridis')

    # visualize_point_cloud_with_colors(primitive_points, distances, colors)

    # primitive_points = np.array(primitive_points)
    # Normalize colors to the range [0, 255]
    # save ctrl_points


    if save_points:
        colors = (colors[:, :3] * 255).astype(np.uint8)
        pointcloud = trimesh.Trimesh(vertices=primitive_points)
        pointcloud.visual.vertex_colors = colors
        pointcloud.export(path_save + '/points/' + shape_name + '_' + str(i) + '_points.ply')

    if save_trimmed:
        new_triangles = trim_mesh_by_distance(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                                        primitive_points, epsilon)
        trimmed_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_triangles)
        trimmed_mesh.visual.face_colors = random_color

        # if the folder exists
        if not os.path.exists(path_save + 'trimmed_surface/'):
            os.makedirs(path_save + 'trimmed_surface/')
            print(f"Created directory: {path_save + 'trimmed_surface/'}")
        trimmed_mesh.export(path_save + 'trimmed_surface/' + shape_name + '_' + str(i) + '_mesh_trimmed.ply')

    # save colored mesh
    if save_colored_mesh:
        mesh.visual.face_colors = random_color
        mesh.export(path_save + 'surface_color/' + shape_name + '_' + str(i) + '_mesh_colored.ply')



    if save_control_polygon:
        pred_grid_points_cpu = ctrl_points.cpu().detach().numpy().squeeze()
        np.save(path_save + 'control_polygon_points/' + shape_name + '_' + str(i) , pred_grid_points_cpu)

        vertices, triangles = generate_triangular_mesh(pred_grid_points_cpu)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.export(path_save + 'control_polygon/' + shape_name + '_' + str(i) + '_cp.ply')


def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")




def nurbs_patch_fitting_pca():
    path = os.path.dirname(os.path.realpath(__file__))

    experiments_path = EXPERIMENTS_PATH
    shape_name = '00873267'

    path_save = os.path.join(experiments_path, shape_name + '/')

    primitives_file = path + '/data/' + shape_name  +  f'/{shape_name}_planar_primitives_detection.vg'
    input_pointcloud_file = path + '/data/' + shape_name + f'/{shape_name}.ply'
    metrics_file = GOCOPP_BASE_PATH + shape_name + f'/{shape_name}_GoCopp_metrics.txt'


    metrics_dict = read_metrics(metrics_file)
    epsilon = 3 * float(metrics_dict['epsilon'] )


    directory_paths = [
        path_save + 'control_polygon',
        path_save + 'control_polygon_points',
        path_save + 'points',
        path_save + 'surface',
        path_save + 'surface_color',
        path_save + 'merged_control_polygon',
        path_save + 'merged_surface'
    ]

    create_directories(directory_paths)

    points, normals, groups, planes = load_primitives_from_vg(primitives_file)


    #network parameters

    net_params = {
        'p': 3,
        'q': 3,
        'n_ctrpts': 4,
        'w_lap': 0.1,
        'w_chamfer': 1,
        'learning_rate': 0.05,
        'samples_res': 100,
        'num_epochs': 20,
        'mod_iter': 21}


    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']

    save_points = True
    save_control_polygon = True
    save_colored_mesh = True

    # chamfer_losses = []
    # laplacian_losses = []

    # #save as ply
    # mesh.export(path + shape_name + '.ply')

    points = np.array(points)
    errors = []


    for i in range(len(groups)):
        print(i)
        primitive_points = points[groups[i]]
        # plot_planes_points(planes[i], primitive_points)

        pca_projection = PCAPlaneProjection(primitive_points, planes[i])
        #
        # # Rotate points and get the new plane
        pca_points, pca_plane, pca_rotation_matrix = pca_projection.rotate_points()
        # plot_planes_points(pca_plane, pca_points)

        grid_points_pca = create_grid_from_plane(pca_plane, pca_points, n_ctrpts_u, n_ctrpts_v)

        # visualize_points(pca_points, grid_points_pca.reshape(-1, 3))

        # local_grid_points, local_points, rotation_matrix, min_vals, max_vals = transform_points_to_local(pca_points, grid_points, pca_plane)
        local_points, min_vals, max_vals = transform_points_to_local(pca_points)

        local_grid_points_trans = create_grid_from_plane(pca_plane, local_points, n_ctrpts_u, n_ctrpts_v)

        # visualize_points(local_points, local_grid_points_trans.reshape(-1,3))

        #create tensor from grid point
        grid_points = torch.tensor(local_grid_points_trans).float().cuda()
        target_vert = torch.tensor(local_points).float().cuda()

        ctrl_points, error = nurbs_fitting(net_params, grid_points, target_vert)
        # errors.append(error.cpu().detach().numpy().astype(float) )

        global_points, ctrl_points_cpu = transform_points_to_global(local_points, ctrl_points, min_vals, max_vals)


        # # Rotate the points back to the original orientation PCA
        original_points = pca_projection.rotate_back(global_points, pca_rotation_matrix)
        grid_points = pca_projection.rotate_back(ctrl_points_cpu, pca_rotation_matrix)


        inp_ctrl_pts = torch.tensor(grid_points).float().cuda()
        # #save the nurbs
        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)
        surface_points = mesh_from_nurbs(path_save + 'surface/' + shape_name + '_' + str(i), n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, sample_size=50)


        tensor_primitive_points  = torch.tensor(primitive_points).float().cuda().unsqueeze(0)
        tensor_surface_points = torch.tensor(surface_points).float().cuda().unsqueeze(0)

        cd_error = chamfer_distance(tensor_surface_points, tensor_primitive_points)

        # print(errors[-1], cd_error)
        errors.append(cd_error[0].cpu().detach().numpy().astype(float))

        save_nurbs_patch(path_save, shape_name, inp_ctrl_pts, primitive_points, i, epsilon, save_points, save_control_polygon, save_colored_mesh)


    print(len(groups))
    #save errors
    errors = np.array(errors)
    np.save(path_save + 'fitting_error.npy', errors)


def fit_nurbs_from_pointcloud(shape_name, primitives_file, path_save, net_params, save_points, save_control_polygon, save_colored_mesh, epsilon):


    # Set output directories
    directory_paths = [
        os.path.join(path_save, 'control_polygon'),
        os.path.join(path_save, 'control_polygon_points'),
        os.path.join(path_save, 'points'),
        os.path.join(path_save, 'surface'),
        os.path.join(path_save, 'surface_color'),
        os.path.join(path_save, 'merged_control_polygon'),
        os.path.join(path_save, 'merged_surface')
    ]
    create_directories(directory_paths)

    # Load primitives data (points, normals, etc.)
    points, normals, groups, planes = load_primitives_from_vg(primitives_file)



    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']

    points = np.array(points)
    errors = []

    for i in range(len(groups)):
        primitive_points = points[groups[i]]
        # plot_planes_points(planes[i], primitive_points)

        pca_projection = PCAPlaneProjection(primitive_points, planes[i])
        #
        # # Rotate points and get the new plane
        pca_points, pca_plane, pca_rotation_matrix = pca_projection.rotate_points()
        # plot_planes_points(pca_plane, pca_points)

        # grid_points_pca = create_grid_from_plane(pca_plane, pca_points, n_ctrpts_u, n_ctrpts_v)

        # visualize_points(pca_points, grid_points_pca.reshape(-1, 3))

        # local_grid_points, local_points, rotation_matrix, min_vals, max_vals = transform_points_to_local(pca_points, grid_points, pca_plane)
        local_points, min_vals, max_vals = transform_points_to_local(pca_points)

        local_grid_points_trans = create_grid_from_plane(pca_plane, local_points, n_ctrpts_u, n_ctrpts_v)

        # visualize_points(local_points, local_grid_points_trans.reshape(-1,3))

        # create tensor from grid point
        grid_points = torch.tensor(local_grid_points_trans).float().cuda()
        target_vert = torch.tensor(local_points).float().cuda()

        ctrl_points, error = nurbs_fitting(net_params, grid_points, target_vert)
        # errors.append(error.cpu().detach().numpy().astype(float) )

        global_points, ctrl_points_cpu = transform_points_to_global(local_points, ctrl_points, min_vals, max_vals)

        # # Rotate the points back to the original orientation PCA
        # original_points = pca_projection.rotate_back(global_points, pca_rotation_matrix)
        grid_points = pca_projection.rotate_back(ctrl_points_cpu, pca_rotation_matrix)

        inp_ctrl_pts = torch.tensor(grid_points).float().cuda()
        # #save the nurbs
        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)
        surface_points = mesh_from_nurbs(path_save + 'surface/' + shape_name + '_' + str(i), n_ctrpts_u, n_ctrpts_v,
                                         inp_ctrl_pts_serial, sample_size=50)

        tensor_primitive_points = torch.tensor(primitive_points).float().cuda().unsqueeze(0)
        tensor_surface_points = torch.tensor(surface_points).float().cuda().unsqueeze(0)

        cd_error = chamfer_distance(tensor_surface_points, tensor_primitive_points)

        # print(errors[-1], cd_error)
        errors.append(cd_error[0].cpu().detach().numpy().astype(float))

        save_nurbs_patch(path_save, shape_name, inp_ctrl_pts, primitive_points, i, epsilon, save_points,
                         save_control_polygon, save_colored_mesh)

    # Save errors to a file
    errors = np.array(errors)
    np.save(os.path.join(path_save, 'fitting_error.npy'), errors)

    print(f"Processed {len(groups)} patches, errors saved at {os.path.join(path_save, 'fitting_error.npy')}.")



if __name__ == '__main__':
    nurbs_patch_fitting_pca()
    # # Set the path to the configuration file
    # path = os.path.dirname(os.path.realpath(__file__))
    # config_file = path + "/config.yaml"
    #
    # # Fit NURBS using parameters from the config file
    # fit_nurbs_from_pointcloud(config_file)
    print('patch fitting...')
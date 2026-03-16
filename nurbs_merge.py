import os

from utils import *
from nurbs_patch_fitting import *

import numpy as np
import trimesh



def average_normal_vector(plane1, plane2):
    """
    Calculate the average normal vector of two planes.

    Parameters:
    - plane1: Coefficients [a, b, c, d] of the first plane equation.
    - plane2: Coefficients [a, b, c, d] of the second plane equation.

    Returns:
    - avg_normal: The average normal vector, normalized.
    """
    normal1 = np.array(plane1[:3])
    normal2 = np.array(plane2[:3])

    # Normalize the normal vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Calculate the average normal vector
    avg_normal = (normal1 + normal2) / 2
    return avg_normal / np.linalg.norm(avg_normal)


def find_middle_point(plane1, plane2):
    """
    Calculate the middle point between two planes.

    Parameters:
    - plane1: Coefficients [a, b, c, d] of the first plane equation.
    - plane2: Coefficients [a, b, c, d] of the second plane equation.

    Returns:
    - midpoint: The midpoint [x, y, z] between the two planes.
    """
    # Define a point on plane1 by setting x and y to 0 and solving for z
    point1 = np.array([0, 0, -plane1[3] / plane1[2]]) if plane1[2] != 0 else \
        (np.array([0, -plane1[3] / plane1[1], 0]) if plane1[1] != 0 else np.array([-plane1[3] / plane1[0], 0, 0]))

    # Define a point on plane2 by setting x and y to 0 and solving for z
    point2 = np.array([0, 0, -plane2[3] / plane2[2]]) if plane2[2] != 0 else \
        (np.array([0, -plane2[3] / plane2[1], 0]) if plane2[1] != 0 else np.array([-plane2[3] / plane2[0], 0, 0]))

    # Calculate the midpoint
    midpoint = (point1 + point2) / 2
    return midpoint







def translate_plane_to_point(plane, point):
    """
    Translates the plane defined by ax + by + cz + d = 0 such that it passes through the point (x0, y0, z0).

    Parameters:
    plane (tuple): Coefficients (a, b, c, d) of the plane equation.
    point (tuple): Coordinates (x0, y0, z0) of the point.

    Returns:
    tuple: Coefficients (a, b, c, d') of the new plane equation.
    """
    a, b, c, d = plane
    x0, y0, z0 = point
    d_prime = -(a * x0 + b * y0 + c * z0)
    return (a, b, c, d_prime)
def average_plane(plane1, plane2):
    """
    Calculate the average plane coefficients of two planes if the angle between them is less than epsilon.

    Parameters:
    - plane1: Coefficients [a, b, c, d] of the first plane equation.
    - plane2: Coefficients [a, b, c, d] of the second plane equation.
    - epsilon: The threshold angle in radians.

    Returns:
    - average_plane_coeffs: The coefficients [a, b, c, d] of the average plane if the angle is less than epsilon, None otherwise.
    """
    # Calculate the normal vectors of the planes
    normal1 = np.array(plane1[:3])
    normal2 = np.array(plane2[:3])

    # Normalize the normal vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Calculate the angle between the normal vectors
    dot_product = np.dot(normal1, normal2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Check if the angle is within the threshold
    # Calculate the average normal vector
    avg_normal = (normal1 + normal2) / 2
    avg_normal = avg_normal / np.linalg.norm(avg_normal)

    # # Find the middle point between the two planes
    # midpoint = find_middle_point(plane1, plane2)
    #
    # # Calculate the new d coefficient using the plane equation ax + by + cz + d = 0
    # avg_d = -np.dot(avg_normal, midpoint)

    avg_d = (plane1[3] + plane2[3]) / 2.0


    return avg_normal[0], avg_normal[1], avg_normal[2], avg_d


def project_points_on_mesh(grid_points, mesh, normal):
    n_x, n_y, _ = grid_points.shape
    updated_grid = np.zeros_like(grid_points)
    ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    for i in range(n_x):
        for j in range(n_y):
            point = grid_points[i, j]
            ray_origin = np.array(point)
            ray_direction = np.array(normal)

            # Perform intersection
            locations, index_ray, index_tri = ray.intersects_location([ray_origin], [ray_direction])

            if locations.shape[0] > 0:
                # Find the closest intersection
                closest_intersection = locations[0]
                updated_grid[i, j] = closest_intersection
            else:
                # No intersection
                updated_grid[i, j] = ray_origin

    return updated_grid


def smooth_border(grid, iterations=10):
    """
    Smooth the borders of a grid of points based on the neighbors.

    Parameters:
    grid (numpy.ndarray): A grid of points with shape (nx, ny, 3).
    iterations (int): Number of iterations for smoothing.

    Returns:
    numpy.ndarray: The smoothed grid of points.
    """
    nx, ny, _ = grid.shape
    smoothed_grid = grid.copy()

    def get_neighbors(x, y):
        neighbors = []
        if x > 0:
            neighbors.append(smoothed_grid[x - 1, y])
        if x < nx - 1:
            neighbors.append(smoothed_grid[x + 1, y])
        if y > 0:
            neighbors.append(smoothed_grid[x, y - 1])
        if y < ny - 1:
            neighbors.append(smoothed_grid[x, y + 1])
        return neighbors

    for _ in range(iterations):
        new_grid = smoothed_grid.copy()

        # Smooth the top and bottom borders
        for y in range(ny):
            # Top border
            neighbors = get_neighbors(0, y)
            if neighbors:
                new_grid[0, y] = np.mean(neighbors, axis=0)

            # Bottom border
            neighbors = get_neighbors(nx - 1, y)
            if neighbors:
                new_grid[nx - 1, y] = np.mean(neighbors, axis=0)

        # Smooth the left and right borders
        for x in range(nx):
            # Left border
            neighbors = get_neighbors(x, 0)
            if neighbors:
                new_grid[x, 0] = np.mean(neighbors, axis=0)

            # Right border
            neighbors = get_neighbors(x, ny - 1)
            if neighbors:
                new_grid[x, ny - 1] = np.mean(neighbors, axis=0)

        smoothed_grid = new_grid

    return smoothed_grid

def project_points_control_polyhedra(points, normal1, normal2, mesh1, mesh2):
    updated_grid = np.zeros_like(points)

    # Project points onto the first mesh
    updated_points_mesh1 = project_points_on_mesh(points, mesh1, normal1)

    # Project points onto the second mesh
    updated_points_mesh2 = project_points_on_mesh(points, mesh2, normal2)

    n_x, n_y, _ = points.shape

    for i in range(n_x):
        for j in range(n_y):
            dist1 = np.linalg.norm(updated_points_mesh1[i, j] - points[i, j])
            dist2 = np.linalg.norm(updated_points_mesh2[i, j] - points[i, j])

            if dist1 > dist2:
                updated_grid[i, j] = updated_points_mesh1[i, j]
            else:
                updated_grid[i, j] = updated_points_mesh2[i, j]

    # updated_grid = smooth_border(updated_grid, iterations=0)

    return updated_grid


def convert_grid_points_to_3d(points, plane_params):
    a, b, c, d = plane_params
    converted_points = []
    for (x, y) in points:
        # Calculate z based on the plane equation
        if c != 0:
            z = - (a * x + b * y + d) / c
        else:
            z = 0  # If normal is parallel to the z-axis, assume z=0

        converted_points.append([x, y, z])

    return np.array(converted_points)

def orient_plane(plane, normals):
    normal = np.array(plane[:3])
    normal = normal / np.linalg.norm(normal)
    avg_normal = np.mean(normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    if np.dot(normal, avg_normal) >= 0:
        return plane
    else:
        return -plane


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    path_save = os.path.dirname(path) + '/nurbs_fitting/data/screw/'

    shape_name = 'screw'
    primitives_file = path + '/data/screw/00947708_planar_primitives_detection.vg'
    points, normals, groups, planes = load_primitives_from_vg(primitives_file)





    #read the control polygons with trimesh
    ctrl_pts_meshes = []
    for i in range(0, len(groups) ):
        ctrlpts_file = path_save + 'control_polygon/' + shape_name + '_' + str(i) + '_cp.ply'
        mesh = trimesh.load(ctrlpts_file)
        ctrl_pts_meshes.append(mesh)



    #network parameters

    net_params = {
        'p': 3,
        'q': 3,
        'n_ctrpts': 4,
        'w_lap': 0.1,
        'w_chamfer': 1,
        'learning_rate': 0.05,
        'samples_res': 100,
        'num_epochs': 5,
        'mod_iter': 1}

    save_points = False
    save_control_polygon = False
    save_colored_mesh = True

    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']
    # n_ctrpts_u = 4
    # n_ctrpts_v = 4

    # chamfer_losses = []
    # laplacian_losses = []

    # #save as ply
    # mesh.export(path + shape_name + '.ply')
    errors = np.load(path_save + 'fitting_error.npy', allow_pickle=True)

    points = np.array(points)
    normals = np.array(normals)
    planes = np.array(planes)
    errors = np.array(errors)

    #orient the plane normals using the point normals
    for i in range(0, len(groups)):
        plane = planes[i]
        normal = np.array(plane[:3])
        normal = normal / np.linalg.norm(normal)

        point_normals = np.array(normals[groups[i]])
        avg_normal = np.mean(point_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        if np.dot(normal, avg_normal) >= 0:
            planes[i]= - planes[i]




    # merge 28 29 sphere patches
    i = 357
    j = 358
    # i = 5
    # j = 6
    epsilon = 10


    merged_points = np.concatenate([points[groups[i]], points[groups[j]]])
    # plot_planes_and_normals_with_points(planes[i], planes[j], points[groups[i]],  points[groups[j]])

    avg_plane = average_plane(planes[i], planes[j])
    # avg_plane = translate_plane_to_point(avg_plane, compute_centroid(merged_points))
    # plot_planes_and_normals_with_points(planes[i], avg_plane, points[groups[i]], merged_points, normals[groups[i]])

    grid_points = create_grid_from_plane(avg_plane, merged_points, n_ctrpts_u, n_ctrpts_v)

    normal = np.array(avg_plane[:3])

    updated_points = project_points_control_polyhedra(grid_points, planes[i][:3], planes[j][:3],  ctrl_pts_meshes[i], ctrl_pts_meshes[j])
    # visualize_projected_grid_points(merged_points, updated_points, merged_points)
    # updated_points = grid_points

    plot_meshes_and_points(ctrl_pts_meshes[i], ctrl_pts_meshes[j], planes[i], planes[j], points[groups[i]], points[groups[j]],
                           str(i), str(j), grid_points, updated_points, avg_plane,
                           normal, merged_points)


    grid_points, merged_points, rotation_matrix, min_vals, max_vals = transform_points_to_local(merged_points,
                                                                                                   updated_points,
                                                                                                   avg_plane)
    grid_points = torch.tensor(grid_points).float().cuda()
    target_vert = torch.tensor(merged_points).float().cuda()

    ctrl_points, error = nurbs_fitting(net_params, grid_points, target_vert)
    error_cpu = error.cpu().detach().numpy().astype(float)

    print(errors[i], errors[j], error_cpu)
    # ctrl_points = grid_points

    merged_points, ctrl_points_cpu = transform_points_to_global(merged_points, ctrl_points, rotation_matrix,
                                                                   min_vals, max_vals)

    inp_ctrl_pts = torch.tensor(ctrl_points_cpu).float().cuda()
    # #save the nurbs
    inp_ctrl_pts = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)

    mesh_from_nurbs(path_save + 'merged_surface/' + shape_name + '_' + str(i) + '_' + str(j), n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts,
                    sample_size=50)

    # save_nurbs_patch(path_save, shape_name, ctrl_points, merged_points, i, save_points, save_control_polygon,
    #                  save_colored_mesh)

if __name__ == '__main__':
    main()
    print('patch merging...')




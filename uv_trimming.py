import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
from scipy.spatial import Delaunay
import os
import json
import trimesh
from utils import read_metrics
from nurbs_patch_fitting import mapping_from_nurbs
from trim_meshes import visualize_points_with_mesh
from nurbs_merge import points_from_nurbs
from utils import visualize_points
from trim_meshes import create_mesh_from_points, filter_long_edges, find_border
from utils import get_random_color_from_colormap
import networkx as nx

def generate_grid(x_range, y_range, resolution):
    """
    Generate a 2D grid between x_range and y_range with a given resolution.
    Ensures all grid points are within the range [0, 1].
    """
    # Generate linearly spaced values, ensuring the range ends at 1
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]
    return points, x, y

def trivial_triangulation(x, y):
    """
    Perform a trivial triangulation of the grid by splitting each square into two triangles.
    """
    triangles = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            # Get the indices of the square's corners
            p1 = j + i * len(y)
            p2 = j + (i + 1) * len(y)
            p3 = j + 1 + i * len(y)
            p4 = j + 1 + (i + 1) * len(y)

            # Divide square into two triangles (p1, p2, p3) and (p3, p2, p4)
            triangles.append([p1, p2, p3])
            triangles.append([p3, p2, p4])

    return np.array(triangles)

def compute_intersections(triangles, points, boundary_lines):
    """
    Compute intersections between triangle edges and the boundary lines.
    """
    intersection_points = []
    for tri in triangles:
        for i in range(3):  # Each edge of the triangle
            edge = LineString([points[tri[i]], points[tri[(i + 1) % 3]]])
            for boundary_line in boundary_lines:
                intersect = edge.intersection(boundary_line)
                if not intersect.is_empty:
                    if "Point" in intersect.geom_type:
                        intersection_points.append(np.array(intersect.coords[0]))
                    elif "MultiPoint" in intersect.geom_type:
                        for point in intersect:
                            intersection_points.append(np.array(point.coords[0]))

    # Remove duplicate points
    return np.unique(np.array(intersection_points), axis=0)

def filter_triangles_inside_boundary(delaunay, all_points, boundary_polygon):
    """
    Filter triangles that are inside the given boundary polygon.
    """
    inside_triangles = []
    for simplex in delaunay.simplices:
        # Compute the centroid of the triangle
        pts = all_points[simplex]
        centroid = np.mean(pts, axis=0)
        # Check if the centroid is inside the boundary
        if boundary_polygon.contains(Point(centroid)):
            inside_triangles.append(simplex)
    return np.array(inside_triangles)

def filter_triangles_inside_multiple_boundary(delaunay, all_points, boundary_polygon_set, idx_max_polygon):
    """
    Filter triangles that are inside the given boundary polygon.
    """
    #pop the max polygon
    max_polygon = boundary_polygon_set.pop(idx_max_polygon)

    inside_triangles = []
    for simplex in delaunay.simplices:
        # Compute the centroid of the triangle
        pts = all_points[simplex]
        centroid = np.mean(pts, axis=0)
        # Check if the centroid is inside the boundary
        if max_polygon.contains(Point(centroid)):
            # if none of the other polygons contain the centroid then add the triangle
            add_triangle = True
            for polygon in boundary_polygon_set:
                if polygon.contains(Point(centroid)):
                    add_triangle = False
                    break
            if add_triangle:
                inside_triangles.append(simplex)
    return np.array(inside_triangles)


def visualize_results(all_points, boundary_coords, intersection_points, inside_triangles):
    """
    Visualize the final result: triangles inside the boundary and key points.
    """
    plt.figure(figsize=(10, 10))

    # Plot the triangles inside the boundary
    for simplex in inside_triangles:
        pts = all_points[simplex]
        plt.fill(*pts.T, edgecolor='black', facecolor='lightgreen', alpha=0.7)

    # Plot the boundary
    boundary_x, boundary_y = zip(*boundary_coords)
    plt.plot(boundary_x, boundary_y, 'r-', label='Boundary', linewidth=2)

    # Plot all points
    plt.scatter(all_points[:, 0], all_points[:, 1], s=10, c="blue", label="Grid + Boundary Points")
    if len(intersection_points) > 0:
        plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=50, c="orange", label="Intersection Points")

    plt.title("Triangles Inside the Boundary")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

def visualize_init_grid(all_points, boundary_coords, inside_triangles):
    """
    Visualize the final result: triangles inside the boundary and key points.
    """
    plt.figure(figsize=(10, 10))

    # Plot the triangles inside the boundary
    for simplex in inside_triangles:
        pts = all_points[simplex]
        plt.fill(*pts.T, edgecolor='black', facecolor='lightgreen', alpha=0.7)

    # Plot the boundary
    boundary_x, boundary_y = zip(*boundary_coords)
    plt.plot(boundary_x, boundary_y, 'r-', label='Boundary', linewidth=2)

    # Plot all points
    plt.scatter(all_points[:, 0], all_points[:, 1], s=10, c="blue", label="Grid + Boundary Points")
    plt.title("Triangles Inside the Boundary")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

def visualize_mesh_triangles(all_points, inside_triangles):
    """
    Visualize the final result: triangles inside the boundary and key points.
    """
    plt.figure(figsize=(10, 10))

    # Plot the triangles inside the boundary
    for simplex in inside_triangles:
        pts = all_points[simplex]
        plt.fill(*pts.T, edgecolor='black', facecolor='lightgreen', alpha=0.7)

    # Plot all points
    plt.scatter(all_points[:, 0], all_points[:, 1], s=10, c="blue", label="Grid + Boundary Points")

    plt.title("grid points and triangles")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


def get_inside_triangulation(boundary_coords, grid_resolution):
    """
    Main function to perform all steps: grid generation, triangulation, and filtering.
    """
    # Convert boundary to Polygon and lines
    boundary_polygon = Polygon(boundary_coords)
    boundary_lines = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in range(len(boundary_coords) - 1)]

    # Generate grid points and trivial triangulation
    grid_points, x, y = generate_grid([0, 1], [0, 1], grid_resolution)
    triangles = trivial_triangulation(x, y)

    visualize_mesh_triangles(grid_points, triangles)

    # Compute intersections
    intersection_points = compute_intersections(triangles, grid_points, boundary_lines)

    # Combine all points: grid points, boundary points, and intersection points
    all_points = np.vstack([grid_points, boundary_coords, intersection_points])

    # Perform final Delaunay triangulation
    final_delaunay = Delaunay(all_points)

    # Filter triangles inside the boundary
    inside_triangles = filter_triangles_inside_boundary(final_delaunay, all_points, boundary_polygon)

    # Visualize the results
    # visualize_results(all_points, boundary_coords, intersection_points, inside_triangles)

    return inside_triangles


def barycentric_coordinates_3d(point, v0, v1, v2):
    """
    Compute barycentric coordinates of a point in 3D space relative to a triangle.
    Args:
        point (ndarray): The point to express in barycentric coordinates.
        v0, v1, v2 (ndarray): The vertices of the triangle in 3D space.
    Returns:
        tuple: Barycentric coordinates (lambda1, lambda2, lambda3).
    """
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0

    # Compute areas using cross products
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    # Determinant of the system
    denom = d00 * d11 - d01 * d01

    # Barycentric coordinates
    lambda2 = (d11 * d20 - d01 * d21) / denom
    lambda3 = (d00 * d21 - d01 * d20) / denom
    lambda1 = 1.0 - lambda2 - lambda3

    return lambda1, lambda2, lambda3


def point_to_uv(point, triangle_3d, triangle_uv):
    """
    Map a point inside a 3D triangle to its corresponding UV coordinates.
    Args:
        point (ndarray): 3D coordinates of the point inside the triangle.
        triangle_3d (ndarray): 3D coordinates of the triangle vertices (3x3).
        triangle_uv (ndarray): 2D UV coordinates of the triangle vertices (3x2).
    Returns:
        ndarray: UV coordinates of the point.
    """
    # Extract triangle vertices in 3D
    v0_3d, v1_3d, v2_3d = triangle_3d

    # Extract UV coordinates of the vertices
    v0_uv, v1_uv, v2_uv = triangle_uv

    # Compute barycentric coordinates of the point
    lambda1, lambda2, lambda3 = barycentric_coordinates_3d(point, v0_3d, v1_3d, v2_3d)

    # Interpolate the UV coordinates using barycentric weights
    uv_point = lambda1 * v0_uv + lambda2 * v1_uv + lambda3 * v2_uv

    return uv_point


def uv_trimming(points_folder, ctrlpoints_folder, output_folder,  n_ctrpts_u, n_ctrpts_v, prefix='_cp', grid_resolution=19):
    """
    Perform UV trimming using the given points and control points.
    """
    for (index, shape_name) in enumerate(os.listdir(ctrlpoints_folder)):
        # if the file is not a mesh file then skip
        if not shape_name.endswith('.ply'):
            continue
        points_name = shape_name.replace(prefix, '_points')

        # Load the mesh and point cloud
        mesh = trimesh.load_mesh(os.path.join(ctrlpoints_folder, shape_name))
        pointcloud = trimesh.load_mesh(os.path.join(points_folder, points_name))
        points = np.asarray(pointcloud.vertices)

        # visualize_points_with_mesh(mesh, [points], ['black'])

        #create nrubs from te controlpolygon points
        inp_ctrl_pts = mesh.vertices
        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)

        # Generate grid points and trivial triangulation
        uv_points, x, y = generate_grid([0, 1], [0, 1], grid_resolution)
        triangles = trivial_triangulation(x, y)

        # visualize_mesh_triangles(uv_points, triangles)

        surface_points = mapping_from_nurbs(n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, uv_points, sample_size=50)

        surface_points = np.array(surface_points)
        surface_mesh = trimesh.Trimesh(vertices=surface_points, faces=triangles)

        # visualize_points(surface_points, inp_ctrl_pts_serial)

        closest_points_inliers, distances, triangle_id = trimesh.proximity.closest_point(surface_mesh, points)
        # visualize_points_with_mesh(surface_mesh, [closest_points_inliers], [ 'black'])

        ball_mesh = create_mesh_from_points(closest_points_inliers)

        # convert to trimesh
        ball_mesh_trimesh = trimesh.Trimesh(np.asarray(ball_mesh.vertices), np.asarray(ball_mesh.triangles))
        # visualize_points_with_mesh(ball_mesh_trimesh, [closest_points_inliers])
        ball_trimesh_refined = filter_long_edges(ball_mesh_trimesh)
        # visualize_points_with_mesh(ball_trimesh_refined, [closest_points_inliers])

        boundary_groups = trimesh.grouping.group_rows(ball_trimesh_refined.edges_sorted, require_count=1)
        boundary_edges = ball_trimesh_refined.edges[boundary_groups]

        projected_inlier_borders = find_border(boundary_edges)

        triangle_id_projected_inlier_borders =  triangle_id[projected_inlier_borders]
        inlier_3d = ball_trimesh_refined.vertices[projected_inlier_borders]


        boundary_coords = []
        for i in range(len(projected_inlier_borders)):
            uv_triangle = uv_points[triangles[triangle_id_projected_inlier_borders[i]]]
            triangle_3d = surface_points[triangles[triangle_id_projected_inlier_borders[i]]]
            inside_point = inlier_3d[i]
            # Compute UV coordinates of the point
            uv_coords = point_to_uv(inside_point, triangle_3d, uv_triangle)
            boundary_coords.append(uv_coords)

        boundary_coords.append(boundary_coords[0])

        # get_inside_triangulation(border_points, grid_resolution)
        boundary_polygon = Polygon(boundary_coords)
        boundary_lines = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in
                          range(len(boundary_coords) - 1)]

        intersection_points = compute_intersections(triangles, uv_points, boundary_lines)

        # Combine all points: grid points, boundary points, and intersection points
        all_points = np.vstack([uv_points, boundary_coords, intersection_points])

        # Perform final Delaunay triangulation
        final_delaunay = Delaunay(all_points)

        # Filter triangles inside the boundary
        inside_triangles = filter_triangles_inside_boundary(final_delaunay, all_points, boundary_polygon)

        # visualize_results(all_points, boundary_coords, intersection_points, inside_triangles)

        all_surface_points = mapping_from_nurbs(n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, all_points)

        #create a mesh
        trimmed_mesh = trimesh.Trimesh(vertices=all_surface_points, faces=inside_triangles)
        # visualize_points_with_mesh(trimmed_mesh, [all_surface_points])
        #save it
        random_color = get_random_color_from_colormap('viridis')
        trimmed_mesh.visual.face_colors = random_color
        trimmed_mesh.export(path_save + shape_name)
        print(path_save + shape_name)

    print("Processing complete.")

    print(f"UV trimming results saved to {output_folder + prefix}.ply")


def find_border_edges(simplices):
    """
    Find the sequence of edges that form the border of the triangulation.
    Args:
        delaunay (Delaunay): Delaunay triangulation object.
    Returns:
        list: List of ordered border paths (each path is a list of vertex indices).
    """
    # Extract all edges from the triangulation
    edges = {}
    for simplex in simplices:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            edge = tuple(sorted((i, j)))  # Store edges as sorted tuples (smallest first)
            edges[edge] = edges.get(edge, 0) + 1  # Count occurrences of each edge

    # Keep only edges that appear in one triangle (border edges)
    border_edges = [edge for edge, count in edges.items() if count == 1]

    # Prepare to traverse edges and construct paths
    border_paths = []  # List of border paths
    visited = set()    # Set of visited edges

    # Traverse edges to construct ordered paths
    for edge in border_edges:
        if edge not in visited:
            path = list(edge)  # Start a new path with the current edge
            visited.add(edge)

            # Traverse forward
            while True:
                found_next = False
                for next_edge in border_edges:
                    if next_edge not in visited:
                        if next_edge[0] == path[-1]:  # Connect forward
                            path.append(next_edge[1])
                            visited.add(next_edge)
                            found_next = True
                            break
                        elif next_edge[1] == path[-1]:  # Connect forward (reverse order)
                            path.append(next_edge[0])
                            visited.add(next_edge)
                            found_next = True
                            break
                if not found_next:
                    break  # Stop when no more edges can be connected

            # Traverse backward
            while True:
                found_prev = False
                for next_edge in border_edges:
                    if next_edge not in visited:
                        if next_edge[1] == path[0]:  # Connect backward
                            path.insert(0, next_edge[0])
                            visited.add(next_edge)
                            found_prev = True
                            break
                        elif next_edge[0] == path[0]:  # Connect backward (reverse order)
                            path.insert(0, next_edge[1])
                            visited.add(next_edge)
                            found_prev = True
                            break
                if not found_prev:
                    break  # Stop when no more edges can be connected

            # Add the constructed path to the list of border paths
            border_paths.append(path)

    return border_paths

def visualize_border(uv_points, delaunay, border_path):
    """
    Visualize the border edges of the triangulation.
    Args:
        uv_points (ndarray): Array of 2D points with shape (N, 2).
        delaunay (Delaunay): Delaunay triangulation object.
        border_path (list): Ordered list of vertices forming the border path.
    """
    plt.figure(figsize=(8, 8))

    # Plot Delaunay triangulation
    plt.triplot(uv_points[:, 0], uv_points[:, 1], delaunay.simplices, color='blue', alpha=0.6, label='Triangles')

    # Plot all UV points
    plt.scatter(uv_points[:, 0], uv_points[:, 1], color='black', label='UV Points')

    # Plot the border path
    border_coords = uv_points[border_path]
    plt.plot(
        np.append(border_coords[:, 0], border_coords[0, 0]),
        np.append(border_coords[:, 1], border_coords[0, 1]),
        'r-', label='Border Path'
    )

    # Annotate border points
    for i, (x, y) in enumerate(border_coords):
        plt.text(x, y, f"B{i}", fontsize=9, color="red")

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("Border Path from Triangulation")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.show()

def compute_edge_lengths(triangles, vertices):
    """
    Compute lengths of edges in a triangular mesh.
    Args:
        triangles (ndarray): Triangles represented as indices of vertices (N, 3).
        vertices (ndarray): Coordinates of the vertices (M, 2 or 3).
    Returns:
        ndarray: Lengths of edges.
        ndarray: Edges as unique pairs of vertex indices.
    """
    edges = []
    lengths = []

    for tri in triangles:
        for i, j in zip(tri, np.roll(tri, -1)):  # Each triangle has 3 edges
            edge = tuple(sorted((i, j)))  # Sort to avoid duplicates
            if edge not in edges:  # Only unique edges
                edges.append(edge)
                lengths.append(np.linalg.norm(vertices[j] - vertices[i]))

    return np.array(edges), np.array(lengths)

def filter_long_edges(triangles, vertices, max_length):
    """
    Filter triangles with edges longer than a given threshold.
    Args:
        triangles (ndarray): Triangles represented as indices of vertices (N, 3).
        vertices (ndarray): Coordinates of the vertices (M, 2 or 3).
        max_length (float): Maximum allowed edge length.
    Returns:
        ndarray: Filtered triangles.
    """
    filtered_triangles = []

    for tri in triangles:
        keep_triangle = True
        for i, j in zip(tri, np.roll(tri, -1)):  # Each triangle has 3 edges
            edge_length = np.linalg.norm(vertices[j] - vertices[i])
            if edge_length > max_length:
                keep_triangle = False
                break
        if keep_triangle:
            filtered_triangles.append(tri)

    return np.array(filtered_triangles)


def group_intersecting_polygons(polygons, bad_polygons):
    G = nx.Graph()

    # Add polygons as nodes
    for i, poly in enumerate(polygons):
        G.add_node(i)

    # Add edges between intersecting polygons
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if not bad_polygons[i] and not bad_polygons[j] and polygons[i].intersects(polygons[j]):
                G.add_edge(i, j)
            # if  polygons[i].intersects(polygons[j]):
            #     G.add_edge(i, j)

    # Find connected components
    groups = [list(component) for component in nx.connected_components(G)]

    # Convert indices back to polygon groups
    return [[polygons[i] for i in group] for group in groups], groups



def uv_trimming2d(points_folder, ctrlpoints_folder, path_save, n_ctrpts_u, n_ctrpts_v, knots_folder='', prefix='_cp', grid_resolution=19, scale_lenght=7):
    """
    Perform UV trimming using the given points and control points.
    """
    for (index, shape_name) in enumerate(os.listdir(ctrlpoints_folder)):
        # if the file is not a mesh file then skip
        if not shape_name.endswith('.ply'):
            continue
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        shape_name_temp = shape_name.replace(prefix, '_uv_trim')
        print(os.path.join(path_save, shape_name_temp))
        if os.path.exists(os.path.join(path_save, shape_name_temp)):
            # print(f"skipping {path_save + shape_name_temp}")
            continue

        points_name = shape_name.replace(prefix, '_points')

        # Load the mesh and point cloud
        mesh = trimesh.load_mesh(os.path.join(ctrlpoints_folder, shape_name))
        pointcloud = trimesh.load(os.path.join(points_folder, points_name), )
        #check if pointcloud type is Scene
        points = np.asarray(pointcloud.vertices)

        # visualize_points_with_mesh(mesh, [points], ['black'])

        #create nrubs from te controlpolygon points
        inp_ctrl_pts = mesh.vertices
        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)

        # Generate grid points and trivial triangulation
        uv_points, x, y = generate_grid([0, 1], [0, 1], grid_resolution)
        triangles = trivial_triangulation(x, y)
        #read the knots from the file
        knots_name = shape_name.replace(prefix + '.ply', '_knots.json')
        knots_file = os.path.join(knots_folder, knots_name)
        knots_u = knots_v = None
        if os.path.exists(knots_file):
            with open(knots_file, 'r') as f:
                json_uv_knots = json.load(f)

            knots_u = np.array(json_uv_knots['knots_u'])
            knots_v = np.array(json_uv_knots['knots_v'])

        # visualize_mesh_triangles(uv_points, triangles)

        surface_points = mapping_from_nurbs(n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, uv_points, knots_u, knots_v, sample_size=50)

        surface_points = np.array(surface_points)
        surface_mesh = trimesh.Trimesh(vertices=surface_points, faces=triangles)

        # visualize_points(surface_points, inp_ctrl_pts_serial)

        closest_points_inliers, distances, triangle_id = trimesh.proximity.closest_point(surface_mesh, points)

        inlier_3d = closest_points_inliers


        inliers_2d_coords = []
        for i in range(len(inlier_3d)):
            uv_triangle = uv_points[triangles[triangle_id[i]]]
            triangle_3d = surface_points[triangles[triangle_id[i]]]
            inside_point = inlier_3d[i]
            # Compute UV coordinates of the point
            uv_coords = point_to_uv(inside_point, triangle_3d, uv_triangle)
            inliers_2d_coords.append(uv_coords)

        inliers_2d_coords = np.array(inliers_2d_coords)

        all_points = np.vstack([uv_points, inliers_2d_coords])
        # visualize_results(all_points, inliers_2d_coords, [], triangles)

        # todo create the 2d boundaries with the projected inliers.
        delaunay_inlier_points = Delaunay(inliers_2d_coords)



        edges, lengths = compute_edge_lengths(delaunay_inlier_points.simplices, points)
        mean_length = np.mean(lengths)

        print(f'Mean edge length: {mean_length}, Median edge length: {np.median(lengths)}')

        # Filter triangles with edges longer than the mean
        filtered_triangles = filter_long_edges(delaunay_inlier_points.simplices, points, max_length=scale_lenght * mean_length)

        border_paths = find_border_edges(filtered_triangles)
        boundary_polygons = []
        boundary_lines_sets = []
        intersection_points_set = []
        border_inlier_points_set = []
        max_area_idx = 0
        max_area = 0
        max_areas = []
        bad_polygons = []
        for i, border_path in enumerate(border_paths):
            border_inlier_points = [inliers_2d_coords[i] for i in border_path]

            border_inlier_points_set.append(border_inlier_points)
            boundary_lines = [LineString([border_inlier_points[i], border_inlier_points[(i + 1) % len(border_inlier_points)]])
                              for i in
                              range(len(border_inlier_points))]
            polygon = Polygon(border_inlier_points)
            if not polygon.is_valid:
                bad_polygons.append(True)
            else:
                bad_polygons.append(False)
            boundary_lines_sets.append(boundary_lines)
            boundary_polygons.append(polygon)
            intersection_points = compute_intersections(triangles, uv_points, boundary_lines)
            intersection_points_set.append(intersection_points)
            max_areas.append(polygon.area)
            if polygon.area > max_area:
                max_area = polygon.area
                max_area_idx = i

        groups, group_id_mapping = group_intersecting_polygons(boundary_polygons, bad_polygons)
        groups_sizes = []
        group_largest_index_per_group = []
        for idx, group in enumerate(groups):
            max_poly_idx = 0
            max_poly_area = 0
            for idp, polygons in enumerate(group):
                if polygons.area > max_poly_area:
                    max_poly_area = polygons.area
                    max_poly_idx = idp
            groups_sizes.append(max_poly_area)
            group_largest_index_per_group.append(max_poly_idx)
            # print(f"Group {idx + 1}: {group}")

        groups_to_keep = [True for group in groups]

        largest_group_size = max(groups_sizes)
        for idx, group in enumerate(groups):
            if groups_sizes[idx] < 0.25 *  largest_group_size:
                groups_to_keep[idx] = False
        groups_to_process = [group for idx, group in enumerate(groups) if groups_to_keep[idx]]
        group_ids_mapping_to_process = [group_id_mapping[idx] for idx, group in enumerate(groups) if groups_to_keep[idx]]
        all_island_meshes = []
        for idx, group_to_process in enumerate(groups_to_process):
            max_poly_idx = 0
            max_poly_area = 0
            for idp, polygons in enumerate(group_to_process):
                if polygons.area > max_poly_area:
                    max_poly_area = polygons.area
                    max_poly_idx = idp
            all_intersection_points = np.vstack([intersection_points_set[i] for i in group_ids_mapping_to_process[idx]])
            all_border_points = np.vstack([border_inlier_points_set[i] for i in group_ids_mapping_to_process[idx]])

            # TODO be a little smarter about what uv points to use (boundingbox)
            all_points = np.vstack([uv_points, all_border_points, all_intersection_points])
            final_delaunay = Delaunay(all_points)
            inside_triangles = filter_triangles_inside_multiple_boundary(final_delaunay, all_points, group_to_process,
                                                                         max_poly_idx)
            # visualize_results(all_points, all_border_points, all_intersection_points, inside_triangles)
            all_surface_points = mapping_from_nurbs(n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, all_points)
            trimmed_mesh = trimesh.Trimesh(vertices=all_surface_points, faces=inside_triangles)
            all_island_meshes.append(trimmed_mesh)
        # all_intersection_points = np.vstack(intersection_points_set)
        # all_border_points = np.vstack(border_inlier_points_set)
        # # [item for sublist in intersection_points_set for item in sublist]
        # all_points = np.vstack([uv_points, all_border_points, all_intersection_points])
        # final_delaunay = Delaunay(all_points)
        #
        # # visualize_results(all_points, all_border_points, intersection_points, final_delaunay.simplices)
        #
        # # Filter triangles inside the boundary
        # inside_triangles = filter_triangles_inside_multiple_boundary(final_delaunay, all_points, boundary_polygons, max_area_idx)
        #
        # # visualize_results(all_points, all_border_points, all_intersection_points, inside_triangles)
        #
        # all_surface_points = mapping_from_nurbs(n_ctrpts_u, n_ctrpts_v, inp_ctrl_pts_serial, all_points)
        #
        # #create a mesh
        # trimmed_mesh = trimesh.Trimesh(vertices=all_surface_points, faces=inside_triangles)
        # # visualize_points_with_mesh(trimmed_mesh, [all_surface_points])

        #concatenate all meshes
        all_meshes = trimesh.util.concatenate(all_island_meshes)
        shape_name = shape_name.replace(prefix, '_uv_trim')

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        print(f'Fixing trimming mesh: {path_save + shape_name}')
        random_color = get_random_color_from_colormap('viridis')
        all_meshes.visual.face_colors = random_color
        all_meshes.export(path_save + shape_name)

    print("Processing complete.")


if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    experiments_path = os.path.join(prefix, 'nurbs_fitting/data/')
    shape_name = 'hand1'
    input_pointcloud_file = path + '/data/' + shape_name + f'/{shape_name}.ply'

    path_save = os.path.join(experiments_path, shape_name + '/')

    metrics_file = os.path.join(path, 'data/' + shape_name + f'/{shape_name}_GoCopp_metrics.txt')
    params_file = path + '/configuration/merge_config.yaml'

    params_dict = read_metrics(params_file)

    metrics_dict = read_metrics(metrics_file)
    epsilon = metrics_dict['epsilon']

    exp_name = params_dict['exp_number']
    epsilon_factor = params_dict['epsilon_factor']

    epsilon_factor = float(epsilon_factor)
    exp_name = float(exp_name)
    epsilon = epsilon_factor * float(epsilon)

    pointcloud = trimesh.load(input_pointcloud_file)
    original_points = np.asarray(pointcloud.vertices)

    n_ctrpts_u = 6
    n_ctrpts_v = 6


    rec_path = path_save + 'merged_surface_color/' + 'mask_' + str(exp_name) + '_theta_' + str(
        epsilon_factor) + '/'

    prefix = '_surfc'
    points_folder = rec_path.replace('merged_surface_color', 'merged_surface_points')
    ctrlpoints_folder = rec_path.replace('merged_surface_color', 'merged_control_polygon')
    output_folder = rec_path.replace('merged_surface_color', 'uv_trimmed_surface_color')
    border_folder = rec_path.replace('merged_surface_color', 'border_points')

    scale_lenght = 5

    uv_trimming2d(points_folder, ctrlpoints_folder, output_folder, n_ctrpts_u, n_ctrpts_v, prefix='_cp',
                grid_resolution=10, scale_lenght=scale_lenght)
    # uv_trimming(points_folder, ctrlpoints_folder, output_folder, n_ctrpts_u, n_ctrpts_v, prefix='_cp', grid_resolution=50)

    # get_inside_triangulation(boundary_coords, grid_resolution=19)
import numpy as np
from scipy.spatial import KDTree
import trimesh
import os
import matplotlib.pyplot as plt
from utils import read_metrics, visualize_points_with_labels
from itertools import cycle
from collections import defaultdict
from scipy.spatial import ConvexHull
import networkx as nx
import pyvista as pv
from utils import   get_random_color_from_colormap
import open3d as o3d
import os
import subprocess

MESH_INTERSECTION_EXECUTABLE = "PATH_TO_MESH_INTERSECTION_EXECUTABLE"

def point_to_line_distance_3d(point, start, end):
    """
    Calculate the perpendicular distance from a point to a line segment in 3D.
    """
    line_vec = np.array(end) - np.array(start)
    point_vec = np.array(point) - np.array(start)

    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)  # Line segment is a single point

    proj = np.dot(point_vec, line_vec) / line_len
    proj_point = np.array(start) + proj * (line_vec / line_len)

    # Clamp projection to the segment
    proj_point = np.clip(proj_point, np.minimum(start, end), np.maximum(start, end))
    return np.linalg.norm(point - proj_point)


def douglas_peucker_3d(points, epsilon):
    """
    Simplify a 3D line using the Douglas-Peucker algorithm.

    Args:
        points (list of list): List of points [x, y, z] representing the 3D line.
        epsilon (float): Threshold distance for simplification.

    Returns:
        list of list: Simplified list of points.
    """
    if len(points) < 3:
        return points  # A line with 2 points cannot be simplified further

    start = points[0]
    end = points[-1]

    # Find the point with the maximum distance from the line segment
    max_dist = 0
    max_index = -1
    for i in range(1, len(points) - 1):
        dist = point_to_line_distance_3d(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursive call for two segments
        first_segment = douglas_peucker_3d(points[:max_index + 1], epsilon)
        second_segment = douglas_peucker_3d(points[max_index:], epsilon)

        # Combine the results, excluding the duplicate middle point
        return first_segment[:-1] + second_segment
    else:
        # If no point is farther than epsilon, return the segment
        return [start, end]


def trim_mesh_by_mesh(vertices, triangles, inlier_points, epsilon):
    """
    Remove triangles whose vertices are within epsilon distance to inlier points.

    Parameters:
    - vertices: np.array of shape (n, 3), the vertices of the mesh
    - triangles: np.array of shape (m, 3), the triangles defined by vertex indices
    - inlier_points: np.array of shape (p, 3), the set of inlier points
    - epsilon: float, the distance threshold for removing triangles

    Returns:
    - new_triangles: np.array of shape (k, 3), the remaining triangles after trimming
    """
    # Build a KDTree for fast nearest-neighbor search
    tree = KDTree(inlier_points)

    # Function to check if a vertex is within epsilon distance to any inlier point
    def is_near_inlier(vertex):
        dist, _ = tree.query(vertex)
        return dist < epsilon

    # Filter triangles: keep triangles where none of the vertices are near inliers
    new_triangles = []
    for tri in triangles:
        v1, v2, v3 = vertices[tri]  # Get vertices of the triangle
        if not (is_near_inlier(v1) or is_near_inlier(v2) or is_near_inlier(v3)):
            new_triangles.append(tri)

    return np.array(new_triangles)

def compute_distances_to_mesh(points, mesh):
    """
    Compute the shortest distance from each point to the surface mesh.

    Parameters:
    - points: A numpy array of shape (n, 3) containing the points (x, y, z).
    - mesh: A trimesh object representing the surface mesh.

    Returns:
    - distances: A numpy array of shape (n,) containing the distances of each point to the mesh.
    """
    # Compute the closest points on the mesh for each point in the point cloud
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, points)
    return distances


def trim_mesh(mesh, points, distance_threshold):
    """
    Trim the triangles of a mesh that do not contain inliers. Inliers are determined by
    the distance from the points to the mesh.

    Parameters:
    - mesh: A trimesh object representing the surface mesh.
    - points: A numpy array of shape (n, 3) containing the points (x, y, z).
    - distance_threshold: A float, points with a distance to the mesh less than this threshold are considered inliers.

    Returns:
    - trimmed_mesh: A new trimesh object with triangles containing no inliers removed.
    """

    # Step 1: Compute distances from points to the mesh
    distances = compute_distances_to_mesh(points, mesh)

    # Step 2: Identify inliers (points within the distance threshold)
    inlier_mask = distances < distance_threshold
    inliers = points[inlier_mask]

    # Step 3: Get the vertex positions of the mesh
    mesh_vertices = mesh.vertices

    # Step 4: Check which triangles contain inliers
    triangles_to_keep = []
    for face in mesh.faces:
        triangle_vertices = mesh_vertices[face]

        # Check if any point in the inliers is within the threshold distance of this triangle
        triangle_inlier = np.any([
            np.linalg.norm(inliers - v, axis=1).min() < distance_threshold
            for v in triangle_vertices
        ])

        if triangle_inlier:
            triangles_to_keep.append(face)

    # Step 5: Create a new mesh with only the triangles that contain inliers
    trimmed_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=np.array(triangles_to_keep))

    return trimmed_mesh

def trim_mesh_by_distance(vertices, triangles, inlier_points, epsilon):
    """
    Remove triangles whose vertices are not within epsilon distance to inlier points.

    Parameters:
    - vertices: np.array of shape (n, 3), the vertices of the mesh
    - triangles: np.array of shape (m, 3), the triangles defined by vertex indices
    - inlier_points: np.array of shape (p, 3), the set of inlier points
    - epsilon: float, the distance threshold for removing triangles

    Returns:
    - new_triangles: np.array of shape (k, 3), the remaining triangles after trimming
    """

    # Build a KDTree for fast nearest-neighbor search
    tree = KDTree(inlier_points)

    # Function to check if a vertex is within epsilon distance to any inlier point
    def is_near_inlier(vertex):
        dist, _ = tree.query(vertex)
        return dist > epsilon

    # Filter triangles: keep triangles where none of the vertices are near inliers
    new_triangles = []
    for tri in triangles:
        v1, v2, v3 = vertices[tri]  # Get vertices of the triangle
        if not (is_near_inlier(v1) or is_near_inlier(v2) or is_near_inlier(v3)):
            new_triangles.append(tri)

    return np.array(new_triangles)

def mask_by_distance_inlier(vertices, triangles, inlier_points, epsilon):
    """
    Filters triangles based on whether any of their vertices are within `epsilon` distance
    of inlier points.

    Args:
        vertices (np.ndarray): An array of shape (n_vertices, 3) representing vertex coordinates.
        triangles (np.ndarray): An array of shape (n_triangles, 3) representing triangle indices.
        inlier_points (np.ndarray): An array of shape (n_inliers, 3) representing inlier point coordinates.
        epsilon (float): Distance threshold for filtering.

    Returns:
        np.ndarray: A boolean mask array of shape (n_triangles,), where True means the triangle
                    is valid (not near any inlier points).
    """
    # Build a KDTree for fast nearest-neighbor search
    tree = KDTree(inlier_points)

    # Function to check if a vertex is within epsilon distance to any inlier point
    def is_near_inlier(vertex):
        dist, _ = tree.query(vertex)
        return dist > epsilon

    # Create a mask for triangles
    mask = np.ones(len(triangles), dtype=bool)  # Initialize all as True (valid)

    # Iterate through triangles and update the mask
    for i, tri in enumerate(triangles):
        v1, v2, v3 = vertices[tri]  # Get vertices of the triangle
        # If any vertex is near an inlier point, mark the triangle as invalid
        if is_near_inlier(v1) or is_near_inlier(v2) or is_near_inlier(v3):
            mask[i] = False

    return mask

def call_trim_meshes(points_folder, mesh_folder, path_save, prefix, epsilon):
    for shape_name in os.listdir(mesh_folder):
        points_name = shape_name.replace(prefix, '_points')
        mesh = trimesh.load_mesh(mesh_folder + shape_name)
        pointcloud = trimesh.load_mesh(points_folder + points_name)
        points = np.asarray(pointcloud.vertices)

        new_triangles = trim_mesh_by_distance(np.asarray(mesh.vertices), np.asarray(mesh.faces), points, epsilon)
        trimmed_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_triangles)
        trimmed_mesh.visual.face_colors = np.asarray(mesh.visual.face_colors[0])

        # if the folder exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # replace orefix with _mesh_trim
        shape_name = shape_name.replace(prefix, '_mesh_trim')
        trimmed_mesh.export(path_save + shape_name)

def visualize_trimmed_mesh(mesh, trimmed_mesh, inlier_points):

    """
    Visualize the meshes and their intersections.

    Args:
        meshes (list[trimesh.Trimesh]): List of meshes.
        intersections (dict): Dictionary of intersection points.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each mesh

    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, alpha=0.3)
    ax.plot_trisurf(trimmed_mesh.vertices[:, 0], trimmed_mesh.vertices[:, 1], trimmed_mesh.vertices[:, 2], triangles=trimmed_mesh.faces, alpha=0.5)

    # Plot intersection points
    ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2], color='red', s=1, label=f"Intersection")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Mesh Intersections")
    plt.legend()
    plt.show()

def visualize_points_with_mesh(mesh, points_list, colors=None):
    """
    Visualize a mesh and a list of points with different colors.

    Args:
        mesh (trimesh.Trimesh): The base mesh to visualize.
        points_list (list[np.ndarray]): List of point arrays, each array of shape (N, 3).
        colors (list[str], optional): List of colors for each point set. If None, default colors are used.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.3,
        color='gray',
        label='Mesh'
    )

    # Generate colors if not provided
    if colors is None:
        colors = cycle(plt.cm.tab10.colors)  # Use tab10 colormap for distinct colors
    else:
        colors = cycle([np.array(color) if isinstance(color, tuple) else color for color in colors])

    # Plot each set of points with its corresponding color
    for idx, points in enumerate(points_list):
        color = next(colors)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            s=10,
            label=f"Points {idx + 1}"
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Mesh and Points Visualization")
    plt.show()


def refine_mesh_to_inliers(mesh, inlier_points, initial_epsilon, final_epsilon, max_iterations=10):
    """
    Iteratively refines a triangular mesh to align its boundaries with inlier points.

    Args:
        mesh (trimesh.Trimesh): The initial triangular mesh.
        inlier_points (np.ndarray): Points considered as inliers.
        initial_epsilon (float): The starting epsilon value for proximity checks. Determines where to not create holes
        final_epsilon (float): The desired epsilon threshold to stop refinement. Determines resolution of the border
        max_iterations (int): Maximum number of iterations for refinement.

    Returns:
        trimesh.Trimesh: The refined mesh with smooth boundaries.
    """
    epsilon = initial_epsilon
    iteration = 0


    # Step 1: Initial trim using inliers
    faces = trim_mesh_by_distance(mesh.vertices, mesh.faces, inlier_points, 2 * epsilon)
    refined_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces)

    # visualize_trimmed_mesh(mesh, refined_mesh, inlier_points)

    # Step 2: Identify initial boundary and non-boundary regions

    while iteration < max_iterations:
        closest_points_inliers, distances, triangle_id = trimesh.proximity.closest_point(refined_mesh, inlier_points)

        boundary_groups = trimesh.grouping.group_rows(refined_mesh.edges_sorted, require_count=1)
        boundary_edges = refined_mesh.edges[boundary_groups]

        faces_with_inlier = np.zeros(len(refined_mesh.faces), dtype=bool)
        faces_with_inlier[triangle_id] = True

        # Identify triangles connected to boundary edges
        boundary_faces_mask = np.zeros(len(refined_mesh.faces), dtype=bool)
        for edge in boundary_edges:
            edge_faces = np.where(
                np.any(np.isin(refined_mesh.faces, edge), axis=1)
            )[0]
            boundary_faces_mask[edge_faces] = True

        faces_with_close_inlier = mask_by_distance_inlier(refined_mesh.vertices, refined_mesh.faces, inlier_points, 2*epsilon)

        boundary_faces = refined_mesh.faces[boundary_faces_mask]
        boundary_mesh = trimesh.Trimesh(vertices=refined_mesh.vertices, faces=boundary_faces)
        # visualize_trimmed_mesh(refined_mesh, boundary_mesh, closest_points_inliers)


        # Keep triangles that are not on the border
        non_border_mask = ~boundary_faces_mask

        # Include border triangles that have inliers
        exception_mask = faces_with_inlier | faces_with_close_inlier

        # Combine the two conditions
        result_mask = non_border_mask | exception_mask

        # Update the refined mesh with only valid triangles
        # refined_mesh = trimesh.Trimesh(vertices=refined_mesh.vertices, faces=refined_mesh.faces[result_mask])
        refined_mesh.update_faces(result_mask)
        refined_mesh.remove_unreferenced_vertices()

        # visualize_trimmed_mesh(prev_refined_mesh, refined_mesh, closest_points_inliers)
        iteration += 1

    return refined_mesh

def find_boundary_points(mesh, inlier_points, initial_epsilon, max_iterations=10):
    """
    Iteratively refines a triangular mesh to align its boundaries with inlier points.

    Args:
        mesh (trimesh.Trimesh): The initial triangular mesh.
        inlier_points (np.ndarray): Points considered as inliers.
        initial_epsilon (float): The starting epsilon value for proximity checks. Determines where to not create holes
        final_epsilon (float): The desired epsilon threshold to stop refinement. Determines resolution of the border
        max_iterations (int): Maximum number of iterations for refinement.

    Returns:
        trimesh.Trimesh: The refined mesh with smooth boundaries.
    """
    epsilon = initial_epsilon
    iteration = 0


    # Step 1: Initial trim using inliers
    faces = trim_mesh_by_distance(mesh.vertices, mesh.faces, inlier_points, epsilon)
    refined_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces)

    # visualize_trimmed_mesh(mesh, refined_mesh, inlier_points)

    # Step 2: Identify initial boundary and non-boundary regions

    while iteration < max_iterations:

        boundary_groups = trimesh.grouping.group_rows(refined_mesh.edges_sorted, require_count=1)
        boundary_edges = refined_mesh.edges[boundary_groups]


        # Identify triangles connected to boundary edges
        boundary_faces_mask = np.zeros(len(refined_mesh.faces), dtype=bool)
        for edge in boundary_edges:
            edge_faces = np.where(
                np.any(np.isin(refined_mesh.faces, edge), axis=1)
            )[0]
            boundary_faces_mask[edge_faces] = True


        boundary_faces = refined_mesh.faces[boundary_faces_mask]
        boundary_mesh = trimesh.Trimesh(vertices=refined_mesh.vertices, faces=boundary_faces)
        boundary_mesh = boundary_mesh.subdivide()

        # Non-boundary triangles remain untouched
        non_boundary_faces = refined_mesh.faces[~boundary_faces_mask]
        non_boundary_mesh = trimesh.Trimesh(vertices=refined_mesh.vertices, faces=non_boundary_faces)
        # visualize_trimmed_mesh(refined_mesh, boundary_mesh, inlier_points)
    #
    # Step 3: Iteratively refine only the boundary mesh


        # Trim the boundary mesh to inliers
        # Subdivide the boundary mesh, e.g. smaller triangles
        # boundary_mesh = boundary_mesh.subdivide()

        # Remove invalid boundary triangles not containing inlier points
        # boundary_centroids = boundary_mesh.triangles_center
        # boundary_distances = np.min(
        #     np.linalg.norm(boundary_centroids[:, None, :] - inlier_points[None, :, :], axis=-1),
        #     axis=1,
        # )

        valid_boundary_faces = trim_mesh_by_distance(boundary_mesh.vertices, boundary_mesh.faces, inlier_points, epsilon)
        # valid_boundary_faces = boundary_mesh.faces[boundary_distances < epsilon]

        # Update the boundary mesh with only valid triangles
        boundary_mesh = trimesh.Trimesh(vertices=boundary_mesh.vertices, faces=valid_boundary_faces)


        # Re calculate boundary and non boundary
        # boundary_groups = trimesh.grouping.group_rows(boundary_mesh.edges_sorted, require_count=1)

        refined_mesh = trimesh.util.concatenate([non_boundary_mesh, boundary_mesh])

        # visualize_trimmed_mesh(refined_mesh, boundary_mesh, inlier_points)
        iteration += 1

    # Step 4: Combine the refined boundary with the original non-boundary mesh

    return refined_mesh


def find_border(edges):
    from collections import defaultdict

    # Build adjacency list
    adj_list = defaultdict(list)
    for a, b in edges:
        adj_list[a].append(b)
        adj_list[b].append(a)

    # Start from any vertex
    start_vertex = edges[0][0]  # Start with the first vertex of the first edge
    visited = set()
    border = []
    current = start_vertex
    prev = None

    # Traverse the edges to find the border
    while current is not None:
        border.append(current)
        visited.add(current)
        next_vertex = None
        for neighbor in adj_list[current]:
            if neighbor != prev and neighbor not in visited:
                next_vertex = neighbor
                break
        prev = current
        current = next_vertex

        if current == start_vertex:  # Close the loop
            break

    return border

def boundary(mesh, close_paths=True):

    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    boundary_edges = mesh.edges[boundary_groups]

    neighbours = defaultdict(lambda: [])
    for v1, v2 in boundary_edges:
        neighbours[v1].append(v2)
        neighbours[v2].append(v1)

    # We now look for all boundary paths by "extracting" one loop at a time. After obtaining a path, we remove its edges
    # from the "boundary_edges" set. The algorithm terminates when all edges have been used.
    boundary_paths = []

    while len(boundary_edges) > 0:
        # Given the set of remaining boundary edges, get one of them and use it to start the current boundary path.
        # In the sequel, v_previous and v_current represent the edge that we are currently processing.
        v_previous, v_current = next(iter(boundary_edges))
        boundary_vertices = [v_previous]

        # Keep iterating until we close the current boundary curve (the "next" vertex is the same as the first one).
        while v_current != boundary_vertices[0]:
            # We grow the path by adding the vertex "v_current".
            boundary_vertices.append(v_current)

            # We now check which is the next vertex to visit.
            v1, v2 = neighbours[v_current]
            if v1 != v_previous:
                v_current, v_previous = v1, v_current
            elif v2 != v_previous:
                v_current, v_previous = v2, v_current
            else:
                # This line should be un-reachable. I am keeping it only to detect bugs in case I made a mistake when
                # designing the algorithm.
                raise RuntimeError(f"Next vertices to visit ({v1=}, {v2=}) are both equal to {v_previous=}.")

        # Close the path (by repeating the first vertex) if needed.
        if close_paths:
            boundary_vertices.append(boundary_vertices[0])

        # "Convert" the vertices from indices to actual Cartesian coordinates.
        boundary_paths.append(boundary_vertices)

        # visualize_trimmed_mesh(mesh, mesh, mesh.vertices[boundary_vertices])

        # Remove all boundary edges that were added to the last path.
        boundary_edges = set(e for e in boundary_edges if e[0] not in boundary_vertices)

    # Return the list of boundary paths.
    return boundary_paths

def visualize_mesh_and_border(edges, border):
    # Create a graph for visualization
    G = nx.Graph()
    G.add_edges_from(edges)

    # Extract coordinates for vertices
    # For simplicity, generate coordinates in a circular layout
    pos = nx.circular_layout(G)

    # Plot the mesh
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)

    # Highlight the border
    border_edges = [(border[i], border[i + 1]) for i in range(len(border) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=border_edges, edge_color="red", width=2.5)

    # Annotate the plot
    plt.title("Mesh and Border Visualization", fontsize=14)
    plt.show()

def call_trim_meshes_refined(points_folder, mesh_folder, path_save, prefix, epsilon):
    for shape_name in os.listdir(mesh_folder):
        points_name = shape_name.replace(prefix, '_points')
        mesh = trimesh.load_mesh(mesh_folder + shape_name)
        pointcloud = trimesh.load_mesh(points_folder + points_name)
        points = np.asarray(pointcloud.vertices)
        print('Trimming mesh:', shape_name)

        trimmed_mesh = refine_mesh_to_inliers(
            mesh=mesh,
            inlier_points=pointcloud.vertices,
            initial_epsilon= epsilon,
            final_epsilon=epsilon,
            max_iterations=2
        )
        random_color = get_random_color_from_colormap('viridis')
        trimmed_mesh.visual.face_colors = random_color

        # if the folder exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # replace orefix with _mesh_trim
        shape_name = shape_name.replace(prefix, '_mesh_trim')
        trimmed_mesh.export(path_save + shape_name)

def extract_border_points(mesh_folder, path_save, prefix):
    all_border_vertices = []
    for shape_name in os.listdir(mesh_folder):

        mesh = trimesh.load_mesh(mesh_folder + shape_name)

        print('extracting borders mesh:', shape_name)

        # vertex_points_list = boundary(mesh, close_paths=True)
        boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        boundary_edges = mesh.edges[boundary_groups]

        vertex_points = find_border(boundary_edges)
        all_border_vertices.append(mesh.vertices[vertex_points])

        # for vertex_points in vertex_points_list:
        #     visualize_trimmed_mesh(mesh, mesh, mesh.vertices[vertex_points])
        # visualize_trimmed_mesh(mesh, mesh, mesh.vertices[vertex_points])

        # new_simp_border = douglas_peucker_3d(mesh.vertices[vertex_points], 0.01)
        # visualize_trimmed_mesh(mesh, mesh, np.asarray(new_simp_border))

        #save the border points as ply file
        #trimesh pointcloud
        pointcloud = trimesh.Trimesh(vertices=mesh.vertices[vertex_points])
        #save the pointcloud


        # if the folder exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # replace orefix with _mesh_trim
        shape_name = shape_name.replace(prefix, 'border')
        pointcloud.export(path_save + shape_name)

    sketch_points  = np.concatenate(all_border_vertices)
    sketch_pointcloud = trimesh.Trimesh(vertices=sketch_points)
    # folder_name = os.path.dirname(os.path.dirname(os.path.dirname(path_save)))
    sketch_pointcloud.export(path_save + f'/{shape_name}_sketch_border.ply')
    print('Saved border points' + path_save + f'/{shape_name}_sketch_border.ply')


def create_cylinder_borders(mesh_folder, path_save, prefix):
    all_border_vertices = []
    for shape_name in os.listdir(mesh_folder):

        input_path = mesh_folder + shape_name
        cylinder_radius = 0.005

        # if the folder exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # replace orefix with _mesh_trim
        shape_name = shape_name.replace(prefix, 'cylinder')
        output_folder = path_save + shape_name
        # Call another program from the terminal
        try:
            subprocess.run([MESH_INTERSECTION_EXECUTABLE,
                            input_path, output_folder, shape_name, str(cylinder_radius)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running the external program: {e}")



def triangular_mesh(points):
    """
    Compute a triangular mesh (surface) from a set of 3D points.

    :param points: (n, 3) numpy array of 3D points.
    :return: ConvexHull object containing the triangular mesh simplices.
    """
    if len(points) < 4:
        raise ValueError("At least four points are required to form a surface mesh.")

    # Compute the convex hull of the points
    hull = ConvexHull(points)
    return hull


def create_triangular_mesh(points):
    """
    Create a triangular surface mesh from a set of 3D points using Trimesh.

    :param points: (n, 3) numpy array of 3D points.
    :return: Trimesh object representing the triangular mesh.
    """
    if len(points) < 4:
        raise ValueError("At least four points are required to create a surface mesh.")

    # Create the convex hull from points
    hull = trimesh.convex.convex_hull(points)
    return hull


def visualize_triangular_mesh(mesh):
    """
    Visualize the triangular surface mesh using Trimesh's built-in viewer.

    :param mesh: Trimesh object representing the triangular surface mesh.
    """
    # Display the mesh in the Trimesh viewer
    mesh.show()

def filter_long_edges(mesh):
    """
    Eliminate edges in the mesh that are longer than the mean edge length.

    :param mesh: Trimesh object representing the triangular surface mesh.
    :return: Trimesh object with long edges removed.
    """
    # Compute edge lengths and edges
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length

    # Compute the mean edge length
    mean_length = np.mean(edge_lengths)

    # Sort edges by length and identify valid edges (shorter than mean)
    valid_edges_mask = edge_lengths <= 5*mean_length
    valid_edges = set(map(tuple, np.sort(edges[valid_edges_mask], axis=1)))

    # Filter faces where all edges are valid
    def face_has_valid_edges(face):
        face_edges = [
            tuple(sorted(face[[0, 1]])),
            tuple(sorted(face[[1, 2]])),
            tuple(sorted(face[[2, 0]])),
        ]
        return all(edge in valid_edges for edge in face_edges)

    valid_faces = [face for face in mesh.faces if face_has_valid_edges(face)]

    # Create and return the new mesh
    return trimesh.Trimesh(vertices=mesh.vertices, faces=valid_faces)

def create_mesh_from_points(points):
    # Convert points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Surface reconstruction using Poisson
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([0.06, 0.15, 0.3])
    )

    # Crop mesh (optional, removes low-density artifacts)
    # bounding_box = pcd.get_axis_aligned_bounding_box()
    # cropped_mesh = mesh.crop(bounding_box)

    return mesh

def remesh_inlier(points_folder, mesh_folder, path_save, prefix):

    for (index, shape_name) in enumerate(os.listdir(mesh_folder)):
        #if the file is not a mesh file then skip
        if not shape_name.endswith('.ply'):
            continue
        points_name = shape_name.replace(prefix, '_points')

        # Load the mesh and point cloud
        mesh = trimesh.load_mesh(os.path.join(mesh_folder, shape_name))
        pointcloud = trimesh.load_mesh(os.path.join(points_folder, points_name))
        points = np.asarray(pointcloud.vertices)

        # Compute the closest points on the mesh to the original points
        closest_points_inliers, distances, triangle_id = trimesh.proximity.closest_point(mesh, points)
        # visualize_points_with_mesh(mesh, [closest_points_inliers], [ 'black'])

        ball_mesh = create_mesh_from_points(closest_points_inliers)

        #convert to trimesh
        poisson_trimesh = trimesh.Trimesh(np.asarray(ball_mesh.vertices), np.asarray(ball_mesh.triangles))
        # visualize_points_with_mesh(poisson_trimesh, [closest_points_inliers])
        poisson_trimesh_refined = filter_long_edges(poisson_trimesh)
        # visualize_points_with_mesh(poisson_trimesh_refined, [points])
        # faces = trim_mesh_by_distance(poisson_trimesh.vertices, poisson_trimesh.faces, closest_points_inliers, 2 * epsilon)
        # poisson_trimesh_refined = trimesh.Trimesh(vertices=mesh.vertices, faces=faces)
        #
        # visualize_points_with_mesh(poisson_trimesh_refined, [points])
        # poisson_trimesh_refined = refine_mesh_to_inliers(poisson_trimesh, closest_points_inliers, epsilon, epsilon, 2)
        # visualize_points_with_mesh(poisson_trimesh_refined, [closest_points_inliers])

        # Create the triangular surface mesh
        # convex_mesh = create_triangular_mesh(closest_points_inliers)
        # visualize_points_with_mesh(convex_mesh, [closest_points_inliers])
        # filtered_mesh = filter_long_edges(convex_mesh)
        # visualize_points_with_mesh(filtered_mesh, [points])

        shape_name = shape_name.replace(prefix, '_remesh_trim')

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        print(f'Fixing trimming mesh: {path_save + shape_name}')
        random_color = get_random_color_from_colormap('viridis')
        poisson_trimesh_refined.visual.face_colors = random_color
        poisson_trimesh_refined.export(path_save + shape_name)

    print("Processing complete.")


if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    experiments_path = os.path.join(prefix, 'nurbs_fitting/data/')
    shape_name = '00873042'
    input_pointcloud_file = path + '/data/' + shape_name + f'/{shape_name}.ply'

    path_save = os.path.join(experiments_path, shape_name + '/')

    metrics_file = os.path.join(path, 'data/' + shape_name + f'/{shape_name}_GoCopp_metrics.txt')
    params_file = path + '/configuration/merge_config_exp_52.0_scale_1.5.yaml'

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


    rec_path = path_save + 'merged_surface_color/' + 'mask_' + str(exp_name) + '_theta_' + str(
        epsilon_factor) + '/'

    prefix = '_surfc'
    points_folder = rec_path.replace('merged_surface_color', 'merged_surface_points')
    output_folder = rec_path.replace('merged_surface_color', 'merged_trimmed_surface_color')
    border_folder = rec_path.replace('merged_surface_color', 'border_points')
    remesh_folder = rec_path.replace('merged_surface_color', 'remeshed_surface_color')
    cylinder_folder = rec_path.replace('merged_surface_color', 'cylinders')
    # remesh_inlier(points_folder, rec_path, remesh_folder, prefix)
    # call_trim_meshes_refined(points_folder, rec_path, output_folder, prefix, epsilon)
    # extract_border_points(remesh_folder, border_folder, prefix='remesh_trim')
    create_cylinder_borders(border_folder, cylinder_folder, prefix='cylinders')
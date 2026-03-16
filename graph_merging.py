import gc

import numpy as np
import trimesh
import shutil
import argparse
import random

import os
import json

from utils import load_primitives_from_vg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
from math import acos, degrees
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import resource
from timeit import default_timer as timer

from utils import (PCAPlaneProjection, read_adjacency_list, get_random_color_from_colormap,
                   read_metrics, clean_merged_folders, visualize_point_sets)
from nurbs_merge import (average_plane, create_grid_from_plane,
                         transform_points_to_local, transform_points_to_global, nurbs_fitting, mesh_from_nurbs, points_from_nurbs,
                         compute_centroid, generate_triangular_mesh, translate_plane_to_point)
from trim_meshes import refine_mesh_to_inliers
from comparison.comparison import chamfer_distance_single_shape_numpy, chamfer_distance_single_shape_kdtree
from quadric_fitting import compute_matrices, solve_taubin_from_matrices


IGNORE_QEM = False
SAVE_IMEDIATE_EACH_NUMBER = 400
SAVE_INTERMEDIATE_PATH = 'PATH_TO_INTERMEDIATE_RESULT_FOLDER'
SAVE_INTERMEDIATE_NAME = 'SAVE_INTERMEDIATE_RESULT_NAME'

class Patch:
    def __init__(self, id, inlier_points=None, plane=None, mesh=None):
        self.id = id
        self.patch_points = np.array(inlier_points) if inlier_points is not None else np.array([]).reshape(0, 3)
        self.plane = plane
        self.mesh = mesh
        self.normal = self.calculate_normal()
        # self.control_points = None
        self.n_ctrpts_u = 4
        self.n_ctrpts_v = 4
        self.control_points = None
        self.knots_u = None
        self.knots_v = None
        self.color = None
        self.M = None #covariance quadric metric
        self.N = None #derivative quadric metric
        self.sum_distance_cd = None

    def add_point(self, point):
        self.patch_points = np.vstack([self.patch_points, point])

    def set_plane(self, plane):
        self.plane = plane

    def set_mesh(self, vertices, faces):
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def calculate_centroid(self):
        if self.patch_points.size == 0:
            return None
        centroid = np.mean(self.patch_points, axis=0)
        return centroid

    def calculate_normal(self):
        if self.plane is None:
            return None
        normal = self.plane[:3]
        return normal / np.linalg.norm(normal)

    def get_surface_points(self):
        surface_points = points_from_nurbs(self.n_ctrpts_u, self.n_ctrpts_v,
                                           self.control_points, sample_size=50)
        return surface_points


    def compute_restricted_distance_cd(self, epsilon):
        surface_points = points_from_nurbs(self.n_ctrpts_u, self.n_ctrpts_v,
                                           self.control_points, sample_size=50)
        cd2 = chamfer_distance_single_shape_numpy(surface_points, self.patch_points, sqrt=True,
                                        one_side=True, reduce=False)

        outliers = self.patch_points[(cd2 > epsilon)]
        #

        if (len(cd2[(cd2 < epsilon)]) != len(cd2)):
            inliers = self.patch_points[(cd2 <= epsilon)]
            self.patch_points = inliers
            visualize_point_sets(surface_points, self.patch_points, outliers)
            print(f"inliers doest match patch points")

        self.sum_distance_cd = np.sum(cd2 [(cd2 < epsilon)], axis=0)
        self.sum_distance_cd = self.sum_distance_cd / epsilon
        return outliers

    def compute_inliers_outliers_bak(self, epsilon, points=None):
        surface_points = points_from_nurbs(self.n_ctrpts_u, self.n_ctrpts_v,
                                           self.control_points, sample_size=50)
        cd2 = chamfer_distance_single_shape_numpy(surface_points, points, sqrt=True,
                                                  one_side=True, reduce=False)
        inliers = points[(cd2 < epsilon)]
        outliers = points[(cd2 >= epsilon)]


        return inliers, outliers, cd2

    def compute_inliers_outliers(self, epsilon, points=None):
        surface_points = points_from_nurbs(self.n_ctrpts_u, self.n_ctrpts_v,
                                           self.control_points, self.knots_u, self.knots_v, sample_size=50)
        cd2 = chamfer_distance_single_shape_kdtree(surface_points, points, sqrt=True,
                                                  one_side=True)
        inliers = points[(cd2 < epsilon)]
        outliers = points[(cd2 >= epsilon)]


        return inliers, outliers, cd2



    def save_surface(self, path_save, merged_surface_str, shape_name, epsilon=0.01, with_color=False, save_points=False, save_trimmed=True, iterator=''):
        if self.control_points is None:
            return

        #replace merged_surface with merged_surface_color
        merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_surface_color')
        merged_points_str = merged_surface_str.replace('merged_surface', 'merged_surface_points')
        merged_trimmed_surface_color = merged_surface_str.replace('merged_surface', 'merged_trimmed_surface_color')


        if not os.path.exists(path_save + merged_surface_str):
            os.makedirs(path_save + merged_surface_str)
        if not os.path.exists(path_save + merged_surface_color_str):
            os.makedirs(path_save + merged_surface_color_str)

        if iterator != '':
            iterator = '' + iterator + '/'
            if not os.path.exists(path_save + merged_surface_str + iterator):
                os.makedirs(path_save + merged_surface_str + iterator)
            if not os.path.exists(path_save + merged_surface_color_str + iterator):
                os.makedirs(path_save + merged_surface_color_str + iterator)

        save_filename = path_save + merged_surface_str + iterator + shape_name + '_' + self.id[:3]
        inp_ctrl_pts_serial = self.control_points.reshape(self.n_ctrpts_u * self.n_ctrpts_v, 3)
        mesh_from_nurbs(save_filename, self.n_ctrpts_u, self.n_ctrpts_v,
                        inp_ctrl_pts_serial,
                        sample_size=50)

        if with_color:
            mesh = trimesh.load(save_filename + '_mesh.off')
            random_color = get_random_color_from_colormap('viridis')
            mesh.visual.face_colors = random_color
            save_color_filename = path_save + merged_surface_color_str + iterator + shape_name + '_' + self.id[:3] + '_surfc.ply'
            mesh.export(save_color_filename)

        if save_points:
            #save as ply

            if not os.path.exists(path_save + merged_points_str):
                os.makedirs(path_save + merged_points_str)

            if iterator != '':
                if not os.path.exists(path_save + merged_points_str + iterator):
                    os.makedirs(path_save + merged_points_str + iterator)

            save_points_filename = path_save + merged_points_str + iterator + shape_name + '_' + self.id[:3] + '_points.ply'

            pointcloud = trimesh.Trimesh(vertices=self.patch_points)
            pointcloud.visual.vertex_colors = self.color
            pointcloud.export(save_points_filename)

        if save_trimmed:

            trimmed_mesh = refine_mesh_to_inliers(
                mesh=mesh,
                inlier_points=self.patch_points,
                initial_epsilon=epsilon,
                final_epsilon=epsilon,
                max_iterations=2
            )
            # visualize_trimmed_mesh(mesh, trimmed_mesh, self.patch_points)

            # new_triangles = trim_mesh_by_distance(np.asarray(mesh.vertices), np.asarray(mesh.faces),
            #                                       self.patch_points, 2 * epsilon)
            # trimmed_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_triangles)
            trimmed_mesh.visual.face_colors = random_color

            # if the folder exists
            if not os.path.exists(path_save + merged_trimmed_surface_color):
                os.makedirs(path_save + merged_trimmed_surface_color)
                print(f"Created directory: {path_save + merged_trimmed_surface_color}")
            if iterator != '':
                if not os.path.exists(path_save + merged_trimmed_surface_color + iterator):
                    os.makedirs(path_save + merged_trimmed_surface_color + iterator)
            trimmed_mesh.export(path_save + merged_trimmed_surface_color + iterator + shape_name + '_' + self.id[:3] + f'_mesh_trim.ply')

        return save_filename, save_color_filename


    def save_control_polygon(self, path_save, merged_surface_str, shape_name, iterator=''):
        merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_control_polygon')
        if not os.path.exists(path_save + merged_surface_color_str):
            os.makedirs(path_save + merged_surface_color_str)

        if iterator != '':
            iterator = '' + iterator + '/'
            if not os.path.exists(path_save + merged_surface_color_str + iterator):
                os.makedirs(path_save + merged_surface_color_str + iterator)

        self.mesh.export(path_save + merged_surface_color_str + iterator + shape_name + '_' + self.id[:3] + '_cp.ply')

    def save_knots(self, path_save, merged_surface_str, shape_name, degree_u, degree_v, iterator=''):
        """
        Saves the NURBS knot vectors and degrees to a JSON file.
        """
        merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_uv_knots')
        if not os.path.exists(path_save + merged_surface_color_str):
            os.makedirs(path_save + merged_surface_color_str)
        # Ensure path exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        if iterator != '':
            iterator_path = os.path.join(path_save, iterator)
            if not os.path.exists(iterator_path):
                os.makedirs(iterator_path)
            path_save = iterator_path

        full_path = path_save + merged_surface_color_str + iterator + shape_name + '_' + self.id[:3] + '_knots.json'

        # The parameters must be lists of floats/ints for JSON serialization
        params = {
            'knots_u': self.knots_u,
            'knots_v': self.knots_v,
            'degree_u': degree_u,
            'degree_v': degree_v
        }

        with open(full_path, 'w') as f:
            json.dump(params, f, indent=4)

    def plot(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh1
        mesh1_faces = self.mesh.vertices[self.mesh.faces]
        ax.add_collection3d(
            Poly3DCollection(mesh1_faces, facecolors='cyan', linewidths=0.1, edgecolors='r', alpha=0.5))

        # Plot the points on each plane
        points1 = self.patch_points
        ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='green', marker='o', label=self.id)

        # Adjust axis limits to fit points
        x_limits = [points1[:, 0].min(), points1[:, 0].max()]
        y_limits = [points1[:, 1].min(), points1[:, 1].max()]
        z_limits = [points1[:, 2].min(), points1[:, 2].max()]

        # Compute the max range for equal scaling
        max_range = np.array([
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0]
        ]).max() / 2.0

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        plt.show()

    def __repr__(self):
        return f"Patch(id={self.id})"

# high fidelity  lamda_pcoverage = 0.4, lambda_fidelity = 0.5, lambda_simplicity = 0.1
class Graph_MP:
    def __init__(self, net_params, shape_name, path_save, epsilon=0.1, n_gocoop_patches=0, outlier_points=None,
                 input_pointcloud=None, l_pcoverage=0.7, l_fidelity=0.1, l_simplicity=0.2, include_outliers=False):
        self.net_params = net_params
        self.shape_name = shape_name
        self.path_save = path_save
        self.graph = None
        self.input_pointcloud = input_pointcloud
        self.sum_total_inliers = 0
        self.epsilon = epsilon
        self.lambda_pcoverage = l_pcoverage
        self.lambda_fidelity = l_fidelity
        self.lambda_simplicity = l_simplicity
        self.initial_primitives = None
        self.sum_all_inliers_number = 0
        self.sum_all_distance_cd = 0
        self.n_gocoop_patches = n_gocoop_patches
        self.n_input_points = len(input_pointcloud)
        self.outlier_points = outlier_points
        self.include_outliers = include_outliers
        self.count_merges = 0

    def compute_U_energy(self, patches=None):
        if patches is None:
            patches = [patch[1]['patch'] for patch in self.graph.nodes(data=True)]

        fidelity, p_coverage, simplicity = self.compute_graph_metrics(patches)
        U = self.lambda_pcoverage * p_coverage + self.lambda_fidelity * fidelity + self.lambda_simplicity * simplicity
        # print(f"U: {U} = {self.lambda_pcoverage} * cov {p_coverage} + {self.lambda_fidelity} * fid {fidelity} + {self.lambda_simplicity} * simp {simplicity}")
        return U

    def compute_graph_metrics(self, patches=None):
        if patches is None:
            patches = [patch[1]['patch'] for patch in self.graph.nodes(data=True)]

        self.compute_sum_cd_count_inliers(patches)
        fidelity = self.sum_all_distance_cd / self.sum_all_inliers_number
        p_coverage = 1 - self.sum_all_inliers_number / self.n_input_points
        simplicity = len(patches) / self.n_gocoop_patches

        return fidelity, p_coverage, simplicity

    def compute_sum_cd_count_inliers(self, patches=None):
        self.sum_all_distance_cd = 0
        self.sum_all_inliers_number = 0
        for patch in patches:
            self.sum_all_distance_cd += patch.sum_distance_cd
            self.sum_all_inliers_number += len(patch.patch_points)


    def angle_between_normals(self, normal1, normal2):
        #normalise normals
        normal1 = normal1 / np.linalg.norm(normal1)
        normal2 = normal2 / np.linalg.norm(normal2)
        cos_theta = np.dot(normal1, normal2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = degrees(acos(cos_theta))
        return angle

    def create_graph(self, patches, adjacency_list, mask_angle=90.0):
        G = nx.Graph()

        # Generate a unique color for each patch using a colormap
        colormap = cm.get_cmap('viridis', len(patches))  # 'hsv' is a good colormap for distinct colors

        for i, patch in enumerate(patches):
            color = colormap(i)
            patch.color = color # Get a unique color for this patch
            G.add_node(i, patch=patch, color=color)


        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                if i < j:  # To avoid duplicate edges in an undirected graph
                    normal1 = patches[i].calculate_normal()
                    normal2 = patches[j].calculate_normal()
                    # coefficients, error = taubin_fit_with_gradient(patches[i].inlier_points)
                    # visualize_implicit_quadric(coefficients, patches[i].inlier_points, error)
                    #
                    # coefficients, error = taubin_fit_with_gradient(patches[j].inlier_points)
                    # visualize_implicit_quadric(coefficients, patches[j].inlier_points, error)
                    #
                    # merged_points = np.concatenate((patches[i].inlier_points, patches[j].inlier_points), axis=0)
                    # coefficients, error = taubin_fit_with_gradient(merged_points)
                    # visualize_implicit_quadric(coefficients, merged_points, error)

                    #initialize quadric matrices
                    M1, N1 = compute_matrices(patches[i].patch_points)
                    patches[i].M = M1
                    patches[i].N = N1
                    M2, N2 = compute_matrices(patches[j].patch_points)
                    patches[j].M = M2
                    patches[j].N = N2
                    M_merged = M1 + M2
                    N_merged = N1 + N2
                    coefficients, error = solve_taubin_from_matrices(M_merged, N_merged)
                    if IGNORE_QEM:
                        error = 0.0

                    # angle = self.angle_between_normals(normal1, normal2)
                    G.add_edge(i, j, weight=error, joinable=True)

        self.graph = G
        return G


    def visualize_adjacent_patches(self, patches):
        """
        Visualize the given patches along with their adjacent patches.
        Plots the patches, their inlier points, normals, and planes in the same plot.

        :param patches: List of Patch objects to visualize along with their adjacent patches.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through the patches and their adjacency
        for patch in patches:
            if patch is None:
                print(f"Patch is None, skipping.")
                continue

            # Plot the main patch
            self._plot_patch(ax, patch, 'red')

            # Check if this patch has neighbors in the graph
            if self.graph:
                patch_node = None
                for node in self.graph.nodes:
                    if self.graph.nodes[node]['patch'] == patch:
                        patch_node = node
                        break

                if patch_node is not None:
                    # Plot all the adjacent patches (neighbors) from the graph
                    neighbors = list(self.graph.neighbors(patch_node))
                    for neighbor in neighbors:
                        neighbor_patch = self.graph.nodes[neighbor]['patch']
                        self._plot_patch(ax, neighbor_patch, 'blue')
                else:
                    print(f"Patch {patch.id} not found in the graph, skipping neighbors.")

        # Set labels and display
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def _plot_patch(self, ax, patch, color = None):
        """
        Helper function to plot a single patch's mesh, inlier points, and normal vector.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if (color is None):
            color = patch.color

        # Plot the mesh if available
        if patch.mesh:
            mesh_faces = patch.mesh.vertices[patch.mesh.faces]
            ax.add_collection3d(
                Poly3DCollection(mesh_faces, facecolors=color, linewidths=0.1, edgecolors='b', alpha=0.5, label=f'Patch {patch.id}'))

        # # Plot the inlier points
        # if patch.inlier_points.size > 0:
        #     ax.scatter(patch.inlier_points[:, 0], patch.inlier_points[:, 1], patch.inlier_points[:, 2],
        #                color=color, marker='o', label=f'Patch {patch.id}')

        # Plot the normal vector
        if patch.normal is not None:
            centroid = patch.calculate_centroid()
            normal_start = centroid
            normal_end = centroid + patch.normal
            ax.quiver(*normal_start, *(normal_end - normal_start), color='black', length=0.1, normalize=True,
                      label=f'Normal {patch.id}')

        # Plot the plane (optional, you can add a function if you want to visualize the plane)

    def plot_close_patches(self, patch_a, patch_b):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh1 (current patch)
        mesh1_faces = patch_a.mesh.vertices[patch_a.mesh.faces]
        ax.add_collection3d(
            Poly3DCollection(mesh1_faces, facecolors='yellow', linewidths=0.1, edgecolors='r', alpha=0.5,
                             label=f'{patch_a.id} Mesh'))

        # Plot mesh2 (other patch)
        mesh2_faces = patch_b.mesh.vertices[patch_b.mesh.faces]
        ax.add_collection3d(
            Poly3DCollection(mesh2_faces, facecolors='orange', linewidths=0.1, edgecolors='b', alpha=0.5,
                             label=f'{patch_b.id} Mesh'))

        # Plot the points for self
        points1 = patch_a.patch_points
        ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='red', marker='o', label=f'{patch_a.id} Points')

        # Plot the points for other_patch
        points2 = patch_b.patch_points
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='blue', marker='o',
                   label=f'{patch_b.id} Points')

        # Adjust axis limits to fit both patches
        all_points = np.vstack([points1, points2])
        x_limits = [all_points[:, 0].min(), all_points[:, 0].max()]
        y_limits = [all_points[:, 1].min(), all_points[:, 1].max()]
        z_limits = [all_points[:, 2].min(), all_points[:, 2].max()]

        # Compute the max range for equal scaling
        max_range = np.array([
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0]
        ]).max() / 2.0

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        plt.show()

    def merge_patches(self, n_ctrpts_u=4, n_ctrpts_v=4, fast_merge_share=0.25):
        if self.graph is None:
            return

        # fast_node_number = len(self.graph.nodes) * (1.0 - fast_merge_share)
        # print(f'Slow node number: {fast_node_number}, total node number: {len(self.graph.nodes)}')

        while True:
            # Find the edge with the minimum weight
            # min_edge = min(self.graph.edges(data=True), key=lambda x: x[2]['weight'], default=None)

            if IGNORE_QEM:
                # Choose random edge to merge, only if weight < inf or explicitly joinable
                eligible_edges = [
                    e for e in self.graph.edges(data=True)
                    if e[2].get('weight', np.inf) < np.inf or e[2].get('joinable') is True
                ]
                min_edge = random.choice(eligible_edges) if eligible_edges else None
            else:
                # Choose edge with minimum weight
                min_edge = min(self.graph.edges(data=True), key=lambda x: x[2]['weight'], default=None)

            # Check if all edges were travelled and no more meres are possible,
            # note that weight of all edges will be inf in that case, always larger or equal to the threshold
            # if two pathces were merged, the number of edges also decreases
            if min_edge is None or min_edge[2]['weight'] >= np.inf or min_edge[2]['joinable'] is False:
                break

            u, v, weight = min_edge
            if self.graph.nodes[u]['patch'] is None or self.graph.nodes[v]['patch'] is None:
                #throw an exception
                raise ValueError("One of the patches is None. Please check the graph creation.")

            # Merge patches u and v
            patch_u = self.graph.nodes[u]['patch']
            patch_v = self.graph.nodes[v]['patch']

            # if len(self.graph.nodes) > fast_node_number:
            #     new_patch = self.merge_fast(patch_u, patch_v)
            # else:
            #     new_patch = self.merge_patch(patch_u, patch_v, n_ctrpts_u, n_ctrpts_v)
            new_patch = self.merge_patch(patch_u, patch_v, n_ctrpts_u, n_ctrpts_v)

            # print(f"Merging patches...", patch_u.id, patch_v.id)

            # visualize control polygons of the two patches
            # self.plot_close_patches(patch_u, patch_v)

            # path_u = new_patch
            # only if patch_u is not None
            if new_patch is None:
                self.graph[u][v]['weight'] = np.inf
                continue

            # Contract the edge
            self.graph = nx.contracted_edge(self.graph, (u, v), self_loops=False)
            self.graph.nodes[u]['patch'] = new_patch
            # print(f"Number of edges: {len(self.graph.edges)}")

            if len(self.graph.nodes) % 10 == 0:
                print(f"Number of nodes: {len(self.graph.nodes)}")

            # Update weights for the new merged node
            for neighbor in list(self.graph.neighbors(u)):
                if self.graph.nodes[neighbor]['patch'] is not None:
                    M_merged = patch_u.M + self.graph.nodes[neighbor]['patch'].M
                    N_merged = patch_u.N + self.graph.nodes[neighbor]['patch'].N
                    coefficients, error = solve_taubin_from_matrices(M_merged, N_merged)
                    if IGNORE_QEM:
                        error = 0.0
                    self.graph[u][neighbor]['weight'] = error

            self.count_merges += 1
            path_save = SAVE_INTERMEDIATE_PATH
            merged_surface_str = SAVE_INTERMEDIATE_NAME
            scaled_epsilon = 0.01

            if SAVE_IMEDIATE_EACH_NUMBER != 0 and self.count_merges % SAVE_IMEDIATE_EACH_NUMBER == 0:
                for patch in self.graph.nodes(data=True):
                    print(patch)
                    patch = patch[1]['patch']
                    if patch.control_points is None:
                        patch = self.fit_patch(patch, n_ctrpts_u, n_ctrpts_v)
                    patch.save_surface(path_save, merged_surface_str, shape_name, epsilon=scaled_epsilon, with_color=True,
                                       save_points=True, iterator=str(self.count_merges))
                    patch.save_control_polygon(path_save, merged_surface_str, shape_name, iterator=str(self.count_merges))

            # self.visualize_graph_3d()
        remaining_patches = [data['patch'] for _, data in self.graph.nodes(data=True) if data['patch'] is not None]


        return remaining_patches

    def visualize_graph_3d(self, highlight_nodes=None):
        if self.graph is None:
            raise ValueError("Graph is not created. Please create the graph before visualization.")

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set default highlight nodes if not provided
        if highlight_nodes is None:
            highlight_nodes = []

        # Ensure highlight_nodes are strings
        highlight_nodes = set(map(str, highlight_nodes))

        # Plot nodes (patches) as surfaces and inlier points
        for node, data in self.graph.nodes(data=True):
            patch = data['patch']
            color = data['color']
            node_id_str = str(node)  # Convert node ID to string for comparison
            is_highlighted = node_id_str in highlight_nodes

            # Determine alpha based on whether the node is highlighted
            alpha = 0.5 if is_highlighted else 0.1

            # Plot inlier points
            inlier_points = patch.patch_points
            if inlier_points.size > 0:
                ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
                           color=color, s=5, alpha=alpha, label=f'Inliers {node_id_str}' if is_highlighted else "")

            # Plot the control polyhedron if this node is highlighted
            if is_highlighted and patch.control_points is not None:
                mesh = trimesh.load(patch.mesh)
                vertices = mesh.vertices
                faces = mesh.faces

                # Plot the mesh by drawing each triangle face+
                for face in faces:
                    triangle = vertices[face]
                    ax.add_collection3d(
                        Poly3DCollection([triangle], color=color, alpha=1, linewidths=0.5, edgecolors='k'))

        # Plot edges (connections between patches)
        for i, j in self.graph.edges():
            node1 = self.graph.nodes[i]['patch']
            node2 = self.graph.nodes[j]['patch']
            ax.plot([node1.patch_points[:, 0].mean(), node2.patch_points[:, 0].mean()],
                    [node1.patch_points[:, 1].mean(), node2.patch_points[:, 1].mean()],
                    [node1.patch_points[:, 2].mean(), node2.patch_points[:, 2].mean()],
                    color='black', linewidth=2)

        # ax.set_title("3D Graph Visualization with Highlighted Nodes")

        ax.set_box_aspect([1, 0.5, 1])
        ax.set_axis_off()

        plt.show()


        # # gif_path = os.path.dirname(os.path.realpath(__file__)) + '/data/tmp/gif/'
        #
        # existing_files = sorted([f for f in os.listdir(gift_path) if f.endswith('.png')])
        #
        # # Determine the next available number
        # if existing_files:
        #     last_file = existing_files[-1]
        #     next_number = int(last_file.split('_')[-1].split('.')[0]) + 1
        # else:
        #     next_number = 1
        #
        # # Save the figure with the next number in the filename
        # filename = os.path.join(gift_path, f'merge_{next_number:03d}.png')
        # plt.savefig(filename)
        # plt.close()  # Close the figure to free up memory

    def fit_patch(self, patch, n_ctrpts_u=4, n_ctrpts_v=4):
        points = patch.patch_points
        plane = patch.plane
        plane = translate_plane_to_point(plane, compute_centroid(points))
        pca_projection = PCAPlaneProjection(points, plane)
        pca_points, pca_plane, pca_rotation_matrix = pca_projection.rotate_points()
        normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        points_local, min_vals, max_vals = transform_points_to_local(pca_points)
        local_grid_points_trans = create_grid_from_plane(pca_plane, points_local, n_ctrpts_u, n_ctrpts_v)
        grid_points = torch.tensor(local_grid_points_trans).float().cuda()
        target_vert = torch.tensor(points_local).float().cuda()
        ctrl_points_local, error, knots_u, knots_v = nurbs_fitting(self.net_params, grid_points, target_vert)
        ctrl_points_local_cpu = ctrl_points_local.cpu().detach().numpy()
        merged_points_global, ctrl_points_cpu = transform_points_to_global(points_local, ctrl_points_local,
                                                                          min_vals, max_vals)
        grid_points = pca_projection.rotate_back(ctrl_points_cpu, pca_rotation_matrix)
        inp_ctrl_pts = torch.tensor(grid_points).float().cuda()
        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)
        pred_grid_points_cpu = inp_ctrl_pts.cpu().detach().numpy().squeeze()
        vertices, triangles = generate_triangular_mesh(pred_grid_points_cpu)
        vertices.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        patch.normal = normal
        patch.n_ctrpts_u = n_ctrpts_u
        patch.n_ctrpts_v = n_ctrpts_v
        patch.mesh = mesh
        patch.control_points = inp_ctrl_pts_serial
        patch.knots_u = knots_u
        patch.knots_v = knots_v

        inliers_patch, outliers_patch, cd2_patch = patch.compute_inliers_outliers(self.epsilon, points)
        patch.sum_distance_cd = (np.sum(cd2_patch[(cd2_patch < self.epsilon)], axis=0)) / self.epsilon
        p_coverage = len(inliers_patch) / len(points)
        outliers = np.concatenate((outliers_patch, self.outlier_points), axis=0)
        # patch.sum_distance_cd = 1

        return patch


    def merge_patch(self, patch_u, patch_v, n_ctrpts_u=4, n_ctrpts_v=4):
        merged_points = np.concatenate((patch_u.patch_points, patch_v.patch_points), axis=0)

        # addiditional_points, remaining_points = filter_and_remove_points(merged_points, self.ignored_points, self.epsilon)
        #
        # if addiditional_points.size > 0:
        #     merged_points = np.concatenate((merged_points, addiditional_points), axis=0)

        # plot_planes_and_normals_with_points(patch_u.plane, patch_v.plane, patch_u.inlier_points, patch_v.inlier_points, addiditional_points)


        avg_plane = average_plane(patch_u.plane, patch_v.plane)
        avg_plane = translate_plane_to_point(avg_plane, compute_centroid(merged_points))

        pca_projection = PCAPlaneProjection(merged_points, avg_plane)

        #
        # # Rotate points and get the new plane
        pca_points, pca_plane, pca_rotation_matrix = pca_projection.rotate_points()
        # plot_planes_points(pca_plane, pca_points)

        # visualize_points(pca_points, grid_points_pca.reshape(-1, 3), " points and grid points PCA space")

        # # Rotate the points back to the original orientation PCA
        # grid_points = create_grid_from_plane(avg_plane, merged_points, n_ctrpts_u, n_ctrpts_v)

        normal = np.array(avg_plane[:3]) / np.linalg.norm(avg_plane[:3])
        # updated_points = project_points_control_polyhedra(grid_points, patch_u.plane[:3], patch_v.plane[:3], patch_u.mesh, patch_v.mesh)

        # visualize_projected_grid_points(pca_points, grid_points, updated_points)

        # plot_meshes_and_points(patch_u.mesh, patch_v.mesh, patch_u.plane, patch_v.plane, patch_u.inlier_points,
        #                        patch_v.inlier_points, patch_u.id, patch_v.id, grid_points_pca, grid_points_pca, avg_plane, normal, pca_points )

        merged_points_local, min_vals, max_vals = transform_points_to_local(pca_points)
        # control_points_list_local, min_vals, max_vals = transform_points_to_local(grid_points_pca.reshape(-1, 3))

        local_grid_points_trans = create_grid_from_plane(pca_plane, merged_points_local, n_ctrpts_u, n_ctrpts_v)

        # visualize_points(merged_points_local, control_points_list_local, " points and grid points local space")

        # grid_points_local = control_points_list_local.reshape(n_ctrpts_u, n_ctrpts_v, 3)


        # merged_points = pca_projection.rotate_back(pca_points, pca_rotation_matrix)
        # grid_points = pca_projection.rotate_back(grid_points, pca_rotation_matrix)

        # visualize_points(merged_points_local, local_grid_points_trans.reshape(-1, 3))

        grid_points = torch.tensor(local_grid_points_trans).float().cuda()
        target_vert = torch.tensor(merged_points_local).float().cuda()


        ctrl_points_local, error, final_knots_u, final_knots_v = nurbs_fitting(self.net_params, grid_points, target_vert)
        # ctrl_points_local_cpu = ctrl_points_local.cpu().detach().numpy()
        # # ctrl_points = grid_points
        # visualize_points(merged_points_local, ctrl_points_local_cpu.reshape(-1, 3))
        # visualize_points(merged_points_local, ctrl_points_local_cpu.reshape(-1, 3), "local")
        merged_points_global, ctrl_points_cpu = transform_points_to_global(merged_points_local, ctrl_points_local,
                                                                    min_vals, max_vals)

        # visualize_points(merged_points_global, ctrl_points_cpu.reshape(-1, 3), "global")
        #undo pca
        # merged_points = pca_projection.rotate_back(merged_points_global, pca_rotation_matrix)
        grid_points = pca_projection.rotate_back(ctrl_points_cpu, pca_rotation_matrix)

        inp_ctrl_pts = torch.tensor(grid_points).float().cuda()
        # error_cpu = error.cpu().detach().numpy().astye(float)

        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)
        # surface_points = points_from_nurbs(n_ctrpts_u, n_ctrpts_v,
        #                                  inp_ctrl_pts_serial, sample_size=50)

        # #save the control polygon
        pred_grid_points_cpu = inp_ctrl_pts.cpu().detach().numpy().squeeze()
        vertices, triangles = generate_triangular_mesh(pred_grid_points_cpu)
        vertices.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        new_patch = Patch(patch_u.id + "_" + patch_v.id, merged_points, avg_plane, mesh)

        new_patch.normal = normal
        # new_patch.control_points = inp_ctrl_pts
        new_patch.n_ctrpts_u = n_ctrpts_u
        new_patch.n_ctrpts_v = n_ctrpts_v
        new_patch.mesh = mesh
        M_merged = patch_u.M + patch_v.M
        N_merged = patch_u.N + patch_v.N
        new_patch.M = M_merged
        new_patch.N = N_merged
        new_patch.control_points = inp_ctrl_pts_serial
        new_patch.knots_u = final_knots_u
        new_patch.knots_v = final_knots_v

        #inliers outliers distribution
        # print(f'Epsilon: {self.epsilon}')

        inliers_patch, outliers_patch, cd2_patch = new_patch.compute_inliers_outliers(self.epsilon, merged_points)
        new_patch.sum_distance_cd = (np.sum(cd2_patch[(cd2_patch < self.epsilon)], axis=0)) / self.epsilon
        # new_patch.sum_distance_cd = 0

        p_coverage = len(inliers_patch) / len(merged_points)
        outliers = np.concatenate((outliers_patch, self.outlier_points), axis=0)


        if self.include_outliers:
            inliers_outliers, outliers_outliers, cd2_outliers = new_patch.compute_inliers_outliers(self.epsilon, self.outlier_points)
            inliers_patch = np.concatenate((inliers_patch, inliers_outliers), axis=0)
            outliers = np.concatenate((outliers_patch, outliers_outliers), axis=0)
            new_patch.sum_distance_cd = ( (np.sum(cd2_patch[(cd2_patch < self.epsilon)], axis=0)) / self.epsilon) + (np.sum(cd2_outliers[(cd2_outliers < self.epsilon)], axis=0)  / (self.epsilon))
            p_coverage = len(inliers_patch) / (len(merged_points) + len(inliers_outliers))


        new_patch.patch_points = inliers_patch

        # print(f"pcoverage: {p_coverage}")
        # print(f"Number of inliers before: {len(merged_points)} and after: {len(inliers_patch)}")

        # if p_coverage < 0.98:
        #     # visualize_point_sets(merged_points, inliers_patch, outliers)
        #     print(f"Error is too high to merge patches {patch_u.id} and {patch_v.id}")
        #     return None


        # sum_previous_fidelity = (patch_u.sum_distance_cd + patch_v.sum_distance_cd) / (
        #             len(patch_u.patch_points) + len(patch_v.patch_points))
        # print(
        #     f"Fidelity of separated sum of patches : {sum_previous_fidelity} and merged {new_patch.sum_distance_cd / len(new_patch.patch_points)}")


        # self.ignored_points = remaining_points
        # print(f"Number of additional points left {len(self.ignored_points)}")

        U = self.compute_U_energy()

        # get the new set of patches
        updated_patches = [patch[1]['patch'] for patch in self.graph.nodes(data=True) if
                           patch[1]['patch'] is not patch_u and patch[1]['patch'] is not patch_v]


        updated_patches.append(new_patch)
        u_new = self.compute_U_energy(updated_patches)

        # visualize_point_sets(merged_points, inliers_patch, outliers)


        delta_U = U - u_new
        # print(f'U before merging {U},  after merging {u_new}, delta U {delta_U}')

        # if p_coverage < 1:
        #

        # patch_u.plot()
        if delta_U < 0 :
            #print the two id patcheds that are not merged with alabel none
            # print(f"Error is too high to merge patches {patch_u.id} and {patch_v.id}")
            return None
        else:
            self.outlier_points = outliers
            # visualize_point_sets(merged_points, inliers_patch, outliers)
            return new_patch

    def merge_fast(self, patch_u, patch_v):
        merged_points = np.concatenate((patch_u.patch_points, patch_v.patch_points), axis=0)
        avg_plane = average_plane(patch_u.plane, patch_v.plane)
        avg_plane = translate_plane_to_point(avg_plane, compute_centroid(merged_points))
        normal = np.array(avg_plane[:3]) / np.linalg.norm(avg_plane[:3])
        new_patch = Patch(patch_u.id + "_" + patch_v.id + "_fast", merged_points, avg_plane)
        new_patch.normal = normal
        M_merged = patch_u.M + patch_v.M
        N_merged = patch_u.N + patch_v.N
        new_patch.M = M_merged
        new_patch.N = N_merged
        new_patch.n_ctrpts_u = patch_u.n_ctrpts_u
        new_patch.n_ctrpts_v = patch_u.n_ctrpts_v
        new_patch.sum_distance_cd = patch_u.sum_distance_cd + patch_v.sum_distance_cd

        return new_patch




def merging_patches(path_save, path, shape_name, primitives_file, adjacency_file, input_pointcloud_file, net_params, metrics_file, params_file,
         exp_name=20.0, epsilon_factor=2, epsilon=0.01, l_pcoverage=0.7, l_fidelity=0.1, l_simplicity=0.2, include_outliers=False):

    epsilon = float(epsilon)

    l_pcoverage = float(l_pcoverage)
    l_fidelity = float(l_fidelity)
    l_simplicity = float(l_simplicity)

    epsilon_factor = float(epsilon_factor)
    exp_name = float(exp_name)

    if include_outliers == 'True':
        include_outliers = True
    else:
        include_outliers = False
    scaled_epsilon = epsilon_factor * epsilon

    points, normals, groups, planes = load_primitives_from_vg(primitives_file)


    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']

    rec_path = path_save + 'merged_trimmed_surface_color/' + 'mask_' + str(exp_name) + '_theta_' + str(
        epsilon_factor) + '/'
    merged_surface_str = 'merged_surface/' + 'mask_' + str(exp_name) + '_theta_' + str(epsilon_factor) + '/'

    # replace merged_surface with merged_surface_color
    merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_surface_color')
    merged_points_str = merged_surface_str.replace('merged_surface', 'merged_surface_points')
    merged_trimmed_surface_color = merged_surface_str.replace('merged_surface', 'merged_trimmed_surface_color')
    merged_control_polygon = merged_surface_str.replace('merged_surface', 'merged_control_polygon')

    if os.path.exists(path_save + merged_surface_color_str):
        shutil.rmtree(path_save + merged_surface_color_str)
    if os.path.exists(path_save + merged_points_str):
        shutil.rmtree(path_save + merged_points_str)
    if os.path.exists(path_save + merged_trimmed_surface_color):
        shutil.rmtree(path_save + merged_trimmed_surface_color)
    if os.path.exists(path_save + merged_control_polygon):
        shutil.rmtree(path_save + merged_control_polygon)

    # If we want to use the whole input set of points
    pointcloud = trimesh.load(input_pointcloud_file)
    original_points = np.asarray(pointcloud.vertices)

    total_n_points = len(original_points)

    points = np.array(points)
    planes = np.array(planes)

    patches = []

    all_grouped_indexes = set(idx for group in groups for idx in group)
    all_possible_indexes = set(range(total_n_points))
    outliers = list(all_possible_indexes - all_grouped_indexes)
    n_outliers = len(outliers)
    outlier_points = original_points[outliers]

    for i in range(0, len(groups)):
        patch = Patch(id=str(i), inlier_points=points[groups[i]], plane=planes[i])
        patch.sum_distance_cd = len(patch.patch_points)
        patches.append(patch)

    print(f'Number of outliers {n_outliers} out of {total_n_points} points')

    adjacency_graph = read_adjacency_list(adjacency_file)
    graph_mp = Graph_MP(net_params, shape_name, path_save, scaled_epsilon, len(groups), outlier_points,
                        original_points, l_pcoverage, l_fidelity, l_simplicity, include_outliers)

    graph_mp.initial_primitives = len(groups)

    graph_mp.create_graph(patches, adjacency_graph, mask_angle=exp_name)

    graph_mp.compute_graph_metrics()
    # Merge patches
    merged_patches = graph_mp.merge_patches(n_ctrpts_u=n_ctrpts_u, n_ctrpts_v=n_ctrpts_v)

    fidelity, p_coverage, simplicity = graph_mp.compute_graph_metrics()

    print(f'Fidelity {fidelity}, p_coverage {p_coverage}, simplicity {simplicity}')

    # # Print the result
    for patch in merged_patches:
        print(patch)
        if patch.control_points is None:
            patch = graph_mp.fit_patch(patch, n_ctrpts_u, n_ctrpts_v)
        patch.save_surface(path_save, merged_surface_str, shape_name, epsilon=scaled_epsilon, with_color=True,
                           save_points=True)
        patch.save_control_polygon(path_save, merged_surface_str, shape_name)

    print(len(merged_patches))
    call_compute_metrics(rec_path, metrics_file, fidelity, p_coverage, simplicity, epsilon_factor, str_type='mesh_trim')

    params_file_copy = os.path.join( path, f'configuration/merge_config_exp_{exp_name}_scale_{epsilon_factor}.yaml')

    # shutil.copyfile(params_file, params_file_copy)

def find_unlisted_indices(groups, total_points):
    """
    Returns a list of indices that are not in the groups.

    Parameters:
        groups (list of list of int): List of lists, each containing indices of points.
        total_points (int): Total number of points in the original set.

    Returns:
        list: Indices of points not listed in groups.
    """
    # Flatten the list of lists into a single set for faster lookup
    listed_indices = {index for group in groups for index in group}

    # Find all indices from 0 to total_points - 1 not in listed_indices
    unlisted_indices = [i for i in range(total_points) if i not in listed_indices]

    return unlisted_indices

def test_main(shape_name = '00873042_lessp'):
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(prefix, 'NURBS_fit/')
    experiments_path = os.path.join( prefix, 'nurbs_fitting/data/')
    print(shape_name)

    path_save = os.path.join(experiments_path, shape_name + '/')

    primitives_file = path + 'data/' + shape_name  +  f'/{shape_name}_planar_primitives_detection.vg'
    adjacency_file = path + 'data/' + shape_name + f'/{shape_name}_adjacency.txt'
    input_pointcloud_file = path + 'data/' + shape_name + f'/{shape_name}.ply'

    metrics_file = path + 'data/' + shape_name + f'/{shape_name}_GoCopp_metrics.txt'
    params_file = path + 'configuration/merge_config_exp_80.0_scale_1.0.yaml'
    # params_file = path + 'configuration/merge_config_exp_53.0_scale_1.0.yaml'
    # params_file = path + 'configuration/merge_config_exp_20.0_scale_1.0.yaml'

    print(os.path.exists(metrics_file))

    metrics_dict = read_metrics(metrics_file)
    params_dict = read_metrics(params_file)

    epsilon = metrics_dict['epsilon']
    epsilon = float(epsilon)
    epsilon = 0.01
    # epsilon = 0.03


    l_pcoverage = params_dict['l_pcoverage']
    l_fidelity = params_dict['l_fidelity']
    l_simplicity = params_dict['l_simplicity']

    exp_name = params_dict['exp_number']
    epsilon_factor = params_dict['epsilon_factor']
    include_outliers = params_dict['include_outliers']

    l_pcoverage = float(l_pcoverage)
    l_fidelity = float(l_fidelity)
    l_simplicity = float(l_simplicity)

    epsilon_factor = float(epsilon_factor)
    exp_name = float(exp_name)

    if include_outliers == 'True':
        include_outliers = True
    else:
        include_outliers = False
    scaled_epsilon = epsilon_factor * epsilon


    points, normals, groups, planes = load_primitives_from_vg(primitives_file)

    net_params = {
        'p': 3,
        'q': 3,
        'n_ctrpts': 4,
        'w_lap': 0.1,
        'w_chamfer': 1,
        'learning_rate': 0.05,
        'samples_res': 100,
        'num_epochs': 20,
        'mod_iter': 21
    }

    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']

    rec_path = path_save + 'merged_trimmed_surface_color/' + 'mask_' + str(exp_name) + '_theta_' + str(
        epsilon_factor) + '/'
    merged_surface_str = 'merged_surface/' + 'mask_' + str(exp_name) + '_theta_' + str(epsilon_factor) + '/'

    # replace merged_surface with merged_surface_color
    merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_surface_color')
    merged_points_str = merged_surface_str.replace('merged_surface', 'merged_surface_points')
    merged_trimmed_surface_color = merged_surface_str.replace('merged_surface', 'merged_trimmed_surface_color')
    merged_control_polygon = merged_surface_str.replace('merged_surface', 'merged_control_polygon')

    if os.path.exists(path_save + merged_surface_color_str):
        shutil.rmtree(path_save + merged_surface_color_str)
    if os.path.exists(path_save + merged_points_str):
        shutil.rmtree(path_save + merged_points_str)
    if os.path.exists(path_save + merged_trimmed_surface_color):
        shutil.rmtree(path_save + merged_trimmed_surface_color)
    if os.path.exists(path_save + merged_control_polygon):
        shutil.rmtree(path_save + merged_control_polygon)

    #If we want to use the whole input set of points
    pointcloud = trimesh.load(input_pointcloud_file)
    original_points = np.asarray(pointcloud.vertices)

    total_n_points = len(original_points)


    points = np.array(points)
    planes = np.array(planes)

    patches = []


    all_grouped_indexes = set(idx for group in groups for idx in group)
    all_possible_indexes = set(range(total_n_points))
    outliers = list(all_possible_indexes - all_grouped_indexes)
    n_outliers = len(outliers)
    outlier_points = original_points[outliers]


    for i in range(0, len(groups)):

        patch = Patch(id=str(i),  inlier_points=points[groups[i]], plane=planes[i])
        patch.sum_distance_cd = len(patch.patch_points)
        patches.append(patch)

    print(f'Number of outliers {n_outliers} out of {total_n_points} points')

    start = timer()

    adjacency_graph = read_adjacency_list(adjacency_file)
    graph_mp = Graph_MP( net_params, shape_name, path_save, scaled_epsilon, len(groups), outlier_points,
                         original_points, l_pcoverage, l_fidelity, l_simplicity, include_outliers)

    graph_mp.initial_primitives = len(groups)

    graph_mp.create_graph(patches, adjacency_graph, mask_angle=exp_name)

    # graph_mp.visualize_graph_3d()

    graph_mp.compute_graph_metrics()
    # Merge patches
    merged_patches = graph_mp.merge_patches(n_ctrpts_u=n_ctrpts_u, n_ctrpts_v=n_ctrpts_v)

    fidelity, p_coverage, simplicity = graph_mp.compute_graph_metrics()

    print(f'Fidelity {fidelity}, p_coverage {p_coverage}, simplicity {simplicity}')

    # graph_mp.visualize_graph_3d()

    end = timer()

    print(f'Elapsed time seconds {end - start}')
    print('memory peak Kilobytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Save timings


    # Print the result
    for patch in merged_patches:
        print(patch)
        if patch.control_points is None:
            patch = graph_mp.fit_patch(patch, n_ctrpts_u, n_ctrpts_v)
        patch.save_surface(path_save, merged_surface_str, shape_name, epsilon = scaled_epsilon, with_color=True, save_points=True)
        patch.save_control_polygon(path_save, merged_surface_str, shape_name)
        patch.save_knots(path_save, merged_surface_str, shape_name, graph_mp.net_params['p'], graph_mp.net_params['q'])

    print(len(merged_patches))
    timings_file = os.path.join(path_save, f'timings_{shape_name}.txt')
    with open(timings_file, 'w') as f:
        f.write(f'Elapsed time seconds {end - start}\n')
        f.write(f'memory peak Kilobytes {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\n')
        f.write(f'Initial patches {len(groups)}\n')
        f.write(f'Final patches {len(merged_patches)}\n')
        f.write(f'number of points {total_n_points}\n')
        f.write('shape name ' + shape_name + '\n')

    params_file_copy = os.path.join(path, f'configuration/merge_config_exp_{exp_name}_scale_{epsilon_factor}.yaml')

    # shutil.copyfile(params_file, params_file_copy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a shapes file.")
    parser.add_argument("-f", type=str, help="input allclouds file.", default='', required=False)
    args = parser.parse_args()
    shape_name = args.f
    if shape_name != '':
        test_main(shape_name)
    else:
        test_main()
    print('patch merging...')

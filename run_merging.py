import os
import json
import random
import shutil
import argparse
import yaml

import numpy as np
import trimesh
import networkx as nx
import torch
import resource
from timeit import default_timer as timer

import matplotlib.cm as cm

# ── local imports (same package) ─────────────────────────────────────────────
from utils import (
    load_primitives_from_vg,
    read_adjacency_list,
    read_metrics,
    get_random_color_from_colormap,
    visualize_point_sets,
    PCAPlaneProjection,
)
from nurbs_merge import (
    mesh_from_nurbs,
    points_from_nurbs,
    nurbs_fitting,
    average_plane,
    create_grid_from_plane,
    transform_points_to_local,
    transform_points_to_global,
    compute_centroid,
    generate_triangular_mesh,
    translate_plane_to_point,
)
from trim_meshes import refine_mesh_to_inliers
from comparison.comparison import chamfer_distance_single_shape_kdtree
from quadric_fitting import compute_matrices, solve_taubin_from_matrices
from uv_trimming import uv_trimming2d

# ── global constants ──────────────────────────────────────────────────────────
IGNORE_QEM = False
SAVE_IMEDIATE_EACH_NUMBER = 400   # 0 = disable intermediate saves


# ─────────────────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────────────────
class Patch:
    def __init__(self, id, inlier_points=None, plane=None, mesh=None):
        self.id = id
        self.patch_points = (
            np.array(inlier_points) if inlier_points is not None
            else np.array([]).reshape(0, 3)
        )
        self.plane = plane
        self.mesh = mesh
        self.normal = self._calculate_normal()
        self.n_ctrpts_u = 4
        self.n_ctrpts_v = 4
        self.control_points = None
        self.knots_u = None
        self.knots_v = None
        self.color = None
        self.M = None   # covariance quadric metric
        self.N = None   # derivative quadric metric
        self.sum_distance_cd = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _calculate_normal(self):
        if self.plane is None:
            return None
        n = self.plane[:3]
        return n / np.linalg.norm(n)

    def calculate_centroid(self):
        if self.patch_points.size == 0:
            return None
        return np.mean(self.patch_points, axis=0)

    def get_surface_points(self):
        return points_from_nurbs(
            self.n_ctrpts_u, self.n_ctrpts_v,
            self.control_points, sample_size=50,
        )

    def compute_inliers_outliers(self, epsilon, points=None):
        surface_points = points_from_nurbs(
            self.n_ctrpts_u, self.n_ctrpts_v,
            self.control_points, self.knots_u, self.knots_v,
            sample_size=50,
        )
        cd2 = chamfer_distance_single_shape_kdtree(
            surface_points, points, sqrt=True, one_side=True,
        )
        inliers = points[cd2 < epsilon]
        outliers = points[cd2 >= epsilon]
        return inliers, outliers, cd2

    # ── save helpers ──────────────────────────────────────────────────────────
    def save_surface(
        self, path_save, merged_surface_str, shape_name,
        epsilon=0.01, with_color=False, save_points=False,
        save_trimmed=True, iterator='',
    ):
        if self.control_points is None:
            return None, None

        merged_surface_color_str = merged_surface_str.replace(
            'merged_surface', 'merged_surface_color'
        )
        merged_points_str = merged_surface_str.replace(
            'merged_surface', 'merged_surface_points'
        )
        merged_trimmed_surface_color = merged_surface_str.replace(
            'merged_surface', 'merged_trimmed_surface_color'
        )

        os.makedirs(path_save + merged_surface_str, exist_ok=True)
        os.makedirs(path_save + merged_surface_color_str, exist_ok=True)

        iter_suffix = ''
        if iterator != '':
            iter_suffix = iterator + '/'
            os.makedirs(path_save + merged_surface_str + iter_suffix, exist_ok=True)
            os.makedirs(path_save + merged_surface_color_str + iter_suffix, exist_ok=True)

        save_filename = (
            path_save + merged_surface_str + iter_suffix
            + shape_name + '_' + self.id[:3]
        )
        inp_ctrl_pts_serial = self.control_points.reshape(
            self.n_ctrpts_u * self.n_ctrpts_v, 3
        )
        mesh_from_nurbs(
            save_filename, self.n_ctrpts_u, self.n_ctrpts_v,
            inp_ctrl_pts_serial, sample_size=50,
        )

        save_color_filename = None
        random_color = None

        if with_color:
            mesh = trimesh.load(save_filename + '_mesh.off')
            random_color = get_random_color_from_colormap('viridis')
            mesh.visual.face_colors = random_color
            save_color_filename = (
                path_save + merged_surface_color_str + iter_suffix
                + shape_name + '_' + self.id[:3] + '_surfc.ply'
            )
            mesh.export(save_color_filename)

        if save_points:
            pts_dir = path_save + merged_points_str + iter_suffix
            os.makedirs(pts_dir, exist_ok=True)
            save_pts_filename = (
                pts_dir + shape_name + '_' + self.id[:3] + '_points.ply'
            )
            pc = trimesh.Trimesh(vertices=self.patch_points)
            pc.visual.vertex_colors = self.color
            pc.export(save_pts_filename)

        if save_trimmed and random_color is not None:
            mesh = trimesh.load(save_filename + '_mesh.off')
            trimmed_mesh = refine_mesh_to_inliers(
                mesh=mesh,
                inlier_points=self.patch_points,
                initial_epsilon=epsilon,
                final_epsilon=epsilon,
                max_iterations=2,
            )
            trimmed_mesh.visual.face_colors = random_color
            trim_dir = path_save + merged_trimmed_surface_color + iter_suffix
            os.makedirs(trim_dir, exist_ok=True)
            trimmed_mesh.export(
                trim_dir + shape_name + '_' + self.id[:3] + '_mesh_trim.ply'
            )

        return save_filename, save_color_filename

    def save_control_polygon(self, path_save, merged_surface_str, shape_name, iterator=''):
        cp_str = merged_surface_str.replace('merged_surface', 'merged_control_polygon')
        iter_suffix = (iterator + '/') if iterator != '' else ''
        os.makedirs(path_save + cp_str + iter_suffix, exist_ok=True)
        self.mesh.export(
            path_save + cp_str + iter_suffix
            + shape_name + '_' + self.id[:3] + '_cp.ply'
        )

    def save_knots(self, path_save, merged_surface_str, shape_name, degree_u, degree_v, iterator=''):
        knots_str = merged_surface_str.replace('merged_surface', 'merged_uv_knots')
        iter_suffix = (iterator + '/') if iterator != '' else ''
        os.makedirs(path_save + knots_str + iter_suffix, exist_ok=True)
        full_path = (
            path_save + knots_str + iter_suffix
            + shape_name + '_' + self.id[:3] + '_knots.json'
        )
        params = {
            'knots_u': self.knots_u,
            'knots_v': self.knots_v,
            'degree_u': degree_u,
            'degree_v': degree_v,
        }
        with open(full_path, 'w') as f:
            json.dump(params, f, indent=4)

    def __repr__(self):
        return f"Patch(id={self.id})"


# ─────────────────────────────────────────────────────────────────────────────
# Graph_MP  (only methods called by test_main kept)
# ─────────────────────────────────────────────────────────────────────────────
class Graph_MP:
    def __init__(
        self, net_params, shape_name, path_save,
        epsilon=0.1, n_gocoop_patches=0,
        outlier_points=None, input_pointcloud=None,
        l_pcoverage=0.7, l_fidelity=0.1, l_simplicity=0.2,
        include_outliers=False,
    ):
        self.net_params = net_params
        self.shape_name = shape_name
        self.path_save = path_save
        self.graph = None
        self.input_pointcloud = input_pointcloud
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

    # ── metrics ───────────────────────────────────────────────────────────────
    def _patches_from_graph(self):
        return [d['patch'] for _, d in self.graph.nodes(data=True)]

    def _sum_cd_count_inliers(self, patches):
        self.sum_all_distance_cd = sum(p.sum_distance_cd for p in patches)
        self.sum_all_inliers_number = sum(len(p.patch_points) for p in patches)

    def compute_graph_metrics(self, patches=None):
        if patches is None:
            patches = self._patches_from_graph()
        self._sum_cd_count_inliers(patches)
        fidelity = self.sum_all_distance_cd / self.sum_all_inliers_number
        p_coverage = 1 - self.sum_all_inliers_number / self.n_input_points
        simplicity = len(patches) / self.n_gocoop_patches
        return fidelity, p_coverage, simplicity

    def compute_U_energy(self, patches=None):
        if patches is None:
            patches = self._patches_from_graph()
        fid, cov, simp = self.compute_graph_metrics(patches)
        return (
            self.lambda_pcoverage * cov
            + self.lambda_fidelity * fid
            + self.lambda_simplicity * simp
        )

    # ── graph creation ────────────────────────────────────────────────────────
    def create_graph(self, patches, adjacency_list, mask_angle=90.0):
        G = nx.Graph()
        colormap = cm.get_cmap('viridis', len(patches))

        for i, patch in enumerate(patches):
            patch.color = colormap(i)
            G.add_node(i, patch=patch, color=patch.color)

        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                if i < j:
                    M1, N1 = compute_matrices(patches[i].patch_points)
                    patches[i].M, patches[i].N = M1, N1
                    M2, N2 = compute_matrices(patches[j].patch_points)
                    patches[j].M, patches[j].N = M2, N2
                    _, error = solve_taubin_from_matrices(M1 + M2, N1 + N2)
                    if IGNORE_QEM:
                        error = 0.0
                    G.add_edge(i, j, weight=error, joinable=True)

        self.graph = G
        return G

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

        return patch

    def merge_patch(self, patch_u, patch_v, n_ctrpts_u=4, n_ctrpts_v=4):
        merged_points = np.concatenate((patch_u.patch_points, patch_v.patch_points), axis=0)

        avg_plane = average_plane(patch_u.plane, patch_v.plane)
        avg_plane = translate_plane_to_point(avg_plane, compute_centroid(merged_points))

        pca_projection = PCAPlaneProjection(merged_points, avg_plane)

        pca_points, pca_plane, pca_rotation_matrix = pca_projection.rotate_points()

        normal = np.array(avg_plane[:3]) / np.linalg.norm(avg_plane[:3])

        merged_points_local, min_vals, max_vals = transform_points_to_local(pca_points)

        local_grid_points_trans = create_grid_from_plane(pca_plane, merged_points_local, n_ctrpts_u, n_ctrpts_v)

        grid_points = torch.tensor(local_grid_points_trans).float().cuda()
        target_vert = torch.tensor(merged_points_local).float().cuda()

        ctrl_points_local, error, final_knots_u, final_knots_v = nurbs_fitting(self.net_params, grid_points, target_vert)

        merged_points_global, ctrl_points_cpu = transform_points_to_global(merged_points_local, ctrl_points_local,
                                                                    min_vals, max_vals)

        grid_points = pca_projection.rotate_back(ctrl_points_cpu, pca_rotation_matrix)

        inp_ctrl_pts = torch.tensor(grid_points).float().cuda()

        inp_ctrl_pts_serial = inp_ctrl_pts.reshape(n_ctrpts_u * n_ctrpts_v, 3)

        pred_grid_points_cpu = inp_ctrl_pts.cpu().detach().numpy().squeeze()
        vertices, triangles = generate_triangular_mesh(pred_grid_points_cpu)
        vertices.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        new_patch = Patch(patch_u.id + "_" + patch_v.id, merged_points, avg_plane, mesh)

        new_patch.normal = normal
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

        inliers_patch, outliers_patch, cd2_patch = new_patch.compute_inliers_outliers(self.epsilon, merged_points)
        new_patch.sum_distance_cd = (np.sum(cd2_patch[(cd2_patch < self.epsilon)], axis=0)) / self.epsilon

        p_coverage = len(inliers_patch) / len(merged_points)
        outliers = np.concatenate((outliers_patch, self.outlier_points), axis=0)

        if self.include_outliers:
            inliers_outliers, outliers_outliers, cd2_outliers = new_patch.compute_inliers_outliers(self.epsilon, self.outlier_points)
            inliers_patch = np.concatenate((inliers_patch, inliers_outliers), axis=0)
            outliers = np.concatenate((outliers_patch, outliers_outliers), axis=0)
            new_patch.sum_distance_cd = ((np.sum(cd2_patch[(cd2_patch < self.epsilon)], axis=0)) / self.epsilon) + (np.sum(cd2_outliers[(cd2_outliers < self.epsilon)], axis=0) / (self.epsilon))
            p_coverage = len(inliers_patch) / (len(merged_points) + len(inliers_outliers))

        new_patch.patch_points = inliers_patch

        U = self.compute_U_energy()

        updated_patches = [patch[1]['patch'] for patch in self.graph.nodes(data=True) if
                           patch[1]['patch'] is not patch_u and patch[1]['patch'] is not patch_v]

        updated_patches.append(new_patch)
        u_new = self.compute_U_energy(updated_patches)

        delta_U = U - u_new

        if delta_U < 0:
            return None
        else:
            self.outlier_points = outliers
            return new_patch

    # ── main merge loop ───────────────────────────────────────────────────────
    def merge_patches(self, n_ctrpts_u=4, n_ctrpts_v=4):
        if self.graph is None:
            return []

        while True:
            if IGNORE_QEM:
                eligible = [
                    e for e in self.graph.edges(data=True)
                    if e[2].get('weight', np.inf) < np.inf
                    or e[2].get('joinable') is True
                ]
                min_edge = random.choice(eligible) if eligible else None
            else:
                min_edge = min(
                    self.graph.edges(data=True),
                    key=lambda x: x[2]['weight'],
                    default=None,
                )

            if (
                min_edge is None
                or min_edge[2]['weight'] >= np.inf
                or min_edge[2]['joinable'] is False
            ):
                break

            u, v, _ = min_edge
            patch_u = self.graph.nodes[u]['patch']
            patch_v = self.graph.nodes[v]['patch']

            if patch_u is None or patch_v is None:
                raise ValueError("Null patch encountered in graph — check graph creation.")

            new_patch = self.merge_patch(patch_u, patch_v, n_ctrpts_u, n_ctrpts_v)

            if new_patch is None:
                self.graph[u][v]['weight'] = np.inf
                continue

            self.graph = nx.contracted_edge(self.graph, (u, v), self_loops=False)
            self.graph.nodes[u]['patch'] = new_patch

            if len(self.graph.nodes) % 10 == 0:
                print(f"  nodes remaining: {len(self.graph.nodes)}")

            # update edge weights for neighbours of merged node
            for nb in list(self.graph.neighbors(u)):
                nb_patch = self.graph.nodes[nb]['patch']
                if nb_patch is not None:
                    _, err = solve_taubin_from_matrices(
                        new_patch.M + nb_patch.M,
                        new_patch.N + nb_patch.N,
                    )
                    if IGNORE_QEM:
                        err = 0.0
                    self.graph[u][nb]['weight'] = err

            self.count_merges += 1

            # optional intermediate saves
            if (
                SAVE_IMEDIATE_EACH_NUMBER != 0
                and self.count_merges % SAVE_IMEDIATE_EACH_NUMBER == 0
            ):
                merged_surface_str = (
                    'merged_surface/mask_'
                    + str(self.count_merges) + '_theta_1.0/'
                )
                for _, node_data in self.graph.nodes(data=True):
                    p = node_data['patch']
                    if p is None:
                        continue
                    if p.control_points is None:
                        p = self.fit_patch(p, n_ctrpts_u, n_ctrpts_v)
                    p.save_surface(
                        self.path_save, merged_surface_str,
                        self.shape_name, epsilon=self.epsilon,
                        with_color=True, save_points=True,
                        iterator=str(self.count_merges),
                    )
                    p.save_control_polygon(
                        self.path_save, merged_surface_str,
                        self.shape_name, iterator=str(self.count_merges),
                    )

        return [
            d['patch']
            for _, d in self.graph.nodes(data=True)
            if d['patch'] is not None
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(shape_name='00873042_lessp'):
    # ── paths ─────────────────────────────────────────────────────────────────
    repo_root = os.path.dirname(os.path.realpath(__file__))
    path           = os.path.join(repo_root, 'data', 'input', '')
    experiments_path = os.path.join(repo_root, 'data', 'output', '')
    path_save      = os.path.join(experiments_path, shape_name, '')
    os.makedirs(path_save, exist_ok=True)

    data_prefix           = os.path.join(path, shape_name, shape_name)
    primitives_file       = data_prefix + '_planar_primitives_detection.vg'
    adjacency_file        = data_prefix + '_adjacency.txt'
    input_pointcloud_file = data_prefix + '.ply'
    metrics_file          = data_prefix + '_GoCopp_metrics.txt'
    params_file           = os.path.join(repo_root, 'configuration', 'merge_config_exp_80.0_scale_1.0.yaml')

    print(shape_name)
    print(os.path.exists(metrics_file))

    # ── load config ───────────────────────────────────────────────────────────
    metrics_dict = read_metrics(metrics_file)
    params_dict  = read_metrics(params_file)

    epsilon          = 0.01                            # override from file
    l_pcoverage      = float(params_dict['l_pcoverage'])
    l_fidelity       = float(params_dict['l_fidelity'])
    l_simplicity     = float(params_dict['l_simplicity'])
    exp_name         = float(params_dict['exp_number'])
    epsilon_factor   = float(params_dict['epsilon_factor'])
    include_outliers = params_dict['include_outliers'] == 'True'
    scaled_epsilon   = epsilon_factor * epsilon

    # ── output folder strings ─────────────────────────────────────────────────
    merged_surface_str = f'merged_surface/mask_{exp_name}_theta_{epsilon_factor}/'
    merged_surface_color_str   = merged_surface_str.replace('merged_surface', 'merged_surface_color')
    merged_points_str          = merged_surface_str.replace('merged_surface', 'merged_surface_points')
    merged_trimmed_surface_color = merged_surface_str.replace('merged_surface', 'merged_trimmed_surface_color')
    merged_control_polygon     = merged_surface_str.replace('merged_surface', 'merged_control_polygon')
    output_folder_uv = merged_surface_str.replace('merged_surface', 'uv_trimmed_surface_color')

    # clean previous outputs
    for folder in [
        merged_surface_color_str, merged_points_str,
        merged_trimmed_surface_color, merged_control_polygon,
    ]:
        full = path_save + folder
        if os.path.exists(full):
            shutil.rmtree(full)

    # ── load data ─────────────────────────────────────────────────────────────
    pointcloud     = trimesh.load(input_pointcloud_file)
    original_points = np.asarray(pointcloud.vertices)
    total_n_points  = len(original_points)

    points, normals, groups, planes = load_primitives_from_vg(primitives_file)
    points = np.array(points)
    planes = np.array(planes)

    # outlier points (not assigned to any primitive)
    all_grouped_indexes = set(idx for group in groups for idx in group)
    all_possible_indexes = set(range(total_n_points))
    outliers = list(all_possible_indexes - all_grouped_indexes)
    n_outliers = len(outliers)
    outlier_points = original_points[outliers]

    patches = []

    for i in range(0, len(groups)):
        patch = Patch(id=str(i), inlier_points=points[groups[i]], plane=planes[i])
        patch.sum_distance_cd = len(patch.patch_points)
        patches.append(patch)

    print(f'Number of outliers {n_outliers} out of {total_n_points} points')

    # ── NURBS params (loaded from config.yaml) ────────────────────────────────
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    net_params = config['network_params']
    n_ctrpts_u = n_ctrpts_v = net_params['n_ctrpts']

    start = timer()

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

    end = timer()

    print(f'Elapsed time seconds {end - start}')
    print('memory peak Kilobytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Print the result
    for patch in merged_patches:
        print(patch)
        if patch.control_points is None:
            patch = graph_mp.fit_patch(patch, n_ctrpts_u, n_ctrpts_v)
        patch.save_surface(path_save, merged_surface_str, shape_name, epsilon=scaled_epsilon, with_color=True, save_points=True)
        patch.save_control_polygon(path_save, merged_surface_str, shape_name)
        patch.save_knots(path_save, merged_surface_str, shape_name, graph_mp.net_params['p'], graph_mp.net_params['q'])

    print(len(merged_patches))

    # write timings
    timings_file = os.path.join(path_save, f'timings_{shape_name}.txt')
    with open(timings_file, 'w') as f:
        f.write(f'Elapsed time seconds {end - start}\n')
        f.write(f'memory peak Kilobytes {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\n')
        f.write(f'Initial patches {len(groups)}\n')
        f.write(f'Final patches {len(merged_patches)}\n')
        f.write(f'number of points {total_n_points}\n')
        f.write(f'shape name {shape_name}\n')

    scale_lenght = 4

    if (not os.path.exists(path_save + output_folder_uv)):
        uv_trimming2d(path_save + merged_points_str, path_save + merged_control_polygon, path_save + output_folder_uv,
                      n_ctrpts_u, n_ctrpts_v, prefix='_cp',
                      grid_resolution=50, scale_lenght=scale_lenght)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NURBS patch merging.')
    parser.add_argument(
        '-f', type=str, default='00873042_lessp',
        help='Shape name (default: 00873042_lessp)',
    )
    args = parser.parse_args()
    run(args.f)
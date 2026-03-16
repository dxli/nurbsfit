import gc

import numpy as np
import torch
import os
from sklearn.neighbors import NearestNeighbors
import trimesh

import matplotlib.pyplot as plt
from PIL import Image

from NURBSDiff.nurbs_eval import SurfEval
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

from torch.autograd.variable import Variable

import time
import glob
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
import yaml
import shutil
import sys



class PlaneProjection:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def project_point(self, point):
        x0, y0, z0 = point
        a, b, c, d = self.a, self.b, self.c, self.d

        if c == 0:
            # For c == 0, the plane is vertical (parallel to the z-axis)
            # Solve for x, y using the equation ax + by + d = 0, and keep z the same
            t = (a * x0 + b * y0 + d) / (a ** 2 + b ** 2)
            x_prime = x0 - a * t
            y_prime = y0 - b * t
            z_prime = z0  # z remains unchanged
        else:
            # General case: Project onto a plane with non-zero a, b, c
            t = (a * x0 + b * y0 + c * z0 + d) / (a ** 2 + b ** 2 + c ** 2)
            x_prime = x0 - a * t
            y_prime = y0 - b * t
            z_prime = z0 - c * t

        return np.array([x_prime, y_prime, z_prime])

    def project_points(self, points):
        return np.array([self.project_point(point) for point in points])

    def visualize(self, points):
        projected_points = self.project_points(points)

        # Create meshgrid for plane plotting
        xx, yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))

        # If c == 0, handle the special case of vertical plane
        if self.c == 0:
            zz = np.zeros_like(xx)  # Plane is parallel to z-axis, we don't plot z dependency
        else:
            zz = (-self.a * xx - self.b * yy - self.d) / self.c

        # Plot the original points and projected points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the plane
        ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.5)

        # Plot original points
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', label='Original Points')

        # Plot projected points
        ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], color='blue', label='Projected Points')

        # Draw lines between original and projected points
        for i in range(len(points)):
            ax.plot([points[i, 0], projected_points[i, 0]],
                    [points[i, 1], projected_points[i, 1]],
                    [points[i, 2], projected_points[i, 2]], 'gray')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Projection of Points onto Plane (c = 0 Case)')

        plt.legend()
        plt.show()


class PCAPlaneProjection:
    def __init__(self, points, original_plane_normal):
        self.points = points
        self.original_plane_normal = [original_plane_normal[0], original_plane_normal[1], original_plane_normal[2]  ]
        self.original_d = original_plane_normal[3]
        self.pca = PCA(n_components=3)
        self.pca.fit(points)

    def rotate_points(self):
        # Use the PCA components to build a rotation matrix
        components = self.pca.components_

        # Rotate points using the PCA components (first 2 axes of PCA become new XY plane)
        rotated_points = np.dot(self.points - np.mean(self.points, axis=0), components.T)

        # The new plane after rotation is aligned with the XY plane
        new_plane_normal = np.array([0, 0, 1])  # New plane normal aligned with the Z-axis
        new_d = -np.mean(rotated_points[:, 2])  # Adjust d to match the mean z of the rotated points
        #translate the points along the normal in opposite direction

        #concat both to return coefs of new roated plane
        new_plane = np.concatenate([new_plane_normal, [new_d]])

        # Return rotated points, new plane, and the rotation matrix (components)
        return rotated_points, new_plane, components

    def rotate_back(self, rotated_points, rotation_matrix):
        # Rotate points back using the inverse of the rotation matrix
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        rotated_back_points = np.dot(rotated_points, inverse_rotation_matrix.T) + np.mean(self.points, axis=0)

        return rotated_back_points

    def visualize(self, rotated_points, original_points, new_plane):
        # Create a meshgrid for both planes (original and rotated)
        xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

        # Original plane: solve for z in terms of x and y using the plane equation ax + by + cz + d = 0
        original_a, original_b, original_c = self.original_plane_normal
        zz_original = (-original_a * xx - original_b * yy - self.original_d) / original_c

        # Rotated plane: z = constant (since it is aligned with XY plane)
        # zz_rotated = np.zeros_like(xx) + new_d  # New plane is aligned with z = 0 approximately

        # Plot the original points and rotated points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # # Plot the original plane
        # ax.plot_surface(xx, yy, zz_original, color='cyan', alpha=0.5, label='Original Plane')
        #
        # # Plot the rotated plane
        # ax.plot_surface(xx, yy, zz_rotated, color='magenta', alpha=0.5, label='New Plane')

        # Plot original points
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], color='green',
                   label='Original Points')

        # Plot original points
        ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], color='red',
                   label='Original Points')

        # Plot rotated points
        ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], color='blue',
                   label='Rotated Points')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Original and Rotated Planes with Points')

        # plt.legend()
        plt.show()

def plot_in_out_pointclouds(inp, out):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(inp[:, 0], inp[:, 1], inp[:, 2], c='r', marker='o')
    ax.scatter(out[:, 0], out[:, 1], out[:, 2], c='b', marker='o')
    plt.show()

def laplacian_loss_unsupervised(output, dist_type="l2"):
    import torch.nn.functional as F

    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    if torch.cuda.is_available():
        filter = Variable(torch.from_numpy(filter)).cuda()
    else:
        filter = Variable(torch.from_numpy(filter))
    # print(output.shape)
    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    # print(laplacian_output.shape)

    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, 1)

        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1))
    dist = torch.mean(dist)
    # num_points = output.shape[1] * output.shape[2] * output.shape[3]
    return dist


def get_dynamic_weights(l_precision, l_recall, epsilon=0.01):
    """
    Computes dynamic weights for the Chamfer distance based on the fitting error.

    Args:
        l_precision (torch.Tensor): The one-way Chamfer loss from surface to target.
        l_recall (torch.Tensor): The one-way Chamfer loss from target to surface.
        epsilon (float): A scale-invariant threshold.

    Returns:
        tuple: A tuple containing the precision and recall weights.
    """
    current_error = (l_precision + l_recall) / 2

    # You would typically get epsilon from your dataset's bounding box.
    # For a normalized point cloud (e.g., in a [-1, 1] box), epsilon would be a small value.
    if current_error.item() > epsilon:
        # Initial stage: prioritize precision
        w_precision = 0.9  # You can tune these values
        w_recall = 0.1
    else:
        # Final stage: prioritize recall
        w_precision = 0.1
        w_recall = 0.9

    return w_precision, w_recall

def nurbs_fitting(net_params, grid_points, target_vert):

    p = net_params['p']
    q = net_params['q']
    n_ctrpts = net_params['n_ctrpts']
    w_lap = net_params['w_lap']
    w_chamfer = net_params['w_chamfer']
    learning_rate = net_params['learning_rate']
    samples_res = net_params['samples_res']
    num_epochs = net_params['num_epochs']
    mod_iter = net_params['mod_iter']


    n_ctrpts_u = n_ctrpts
    n_ctrpts_v = n_ctrpts

    sample_size_u = samples_res
    sample_size_v = samples_res



    tgt_cpu = target_vert.detach().cpu().numpy().squeeze()
    # grid_points_cpu = grid_points.detach().cpu().numpy().squeeze()

    # visualize_input_grid_points(grid_points_cpu, tgt_cpu)



    knot_int_u = torch.nn.Parameter(torch.ones(n_ctrpts_u - p).unsqueeze(0).cuda(), requires_grad=True)
    knot_int_v = torch.nn.Parameter(torch.ones(n_ctrpts_v - q).unsqueeze(0).cuda(), requires_grad=True)
    weights = torch.nn.Parameter(torch.ones(1, n_ctrpts_u, n_ctrpts_v, 1).float().cuda(), requires_grad=True)
    layer = SurfEval(n_ctrpts_u, n_ctrpts_v, dimension=3, p=p, q=q, out_dim_u=sample_size_u,
                     out_dim_v=sample_size_v, method='tc', dvc='cuda').cuda()

    inp_ctrl_pts = grid_points.unsqueeze(0).contiguous().float().cuda()

    inp_ctrl_pts.requires_grad = True

    opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=learning_rate)
    opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.1, verbose=True, min_lr=1e-5,
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel',
                                                              cooldown=0,
                                                              )
    lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=5, factor=0.1, verbose=True, min_lr=1e-5,
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel',
                                                              cooldown=0, )

    pbar = tqdm(range(num_epochs), disable=True)
    time1 = time.time()

    gc.collect()

    for i in pbar:


        knot_rep_p_0 = torch.zeros(1, p + 1).cuda()
        knot_rep_p_1 = torch.zeros(1, p).cuda()
        knot_rep_q_0 = torch.zeros(1, q + 1).cuda()
        knot_rep_q_1 = torch.zeros(1, q).cuda()


        with torch.no_grad():
            def closure():
                if i % 100 < 30:
                    opt1.zero_grad()
                else:
                    opt2.zero_grad()
                # opt1.zero_grad()

                out = layer((
                    torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1),
                    torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))
                loss = 0

                loss_laplacian = laplacian_loss_unsupervised(inp_ctrl_pts)
                out = out.reshape(sample_size_u, sample_size_v, 3)

                tgt = target_vert.unsqueeze(0)


                out = out.reshape(1, sample_size_u * sample_size_v, 3)

                loss_chamfer, _ = chamfer_distance(out, tgt)
                ########################################
                # Get the precision term (surface to target)
                # l_precision, _ = chamfer_distance(
                #     x=out,
                #     y=tgt,
                #     single_directional=True,
                #     point_reduction='mean'
                # )
                #
                # # Get the recall term (target to surface)
                # l_recall, _ = chamfer_distance(
                #     x=tgt,
                #     y=out,
                #     single_directional=True,
                #     point_reduction='mean'
                # )
                #
                # w_precision, w_recall = get_dynamic_weights(l_precision, l_recall)
                # Combine them into the final Chamfer loss
                # loss_chamfer = w_precision * l_precision + w_recall * l_recall

                ####################################
                # out = None
                # tgt = None
                loss = w_chamfer * loss_chamfer + w_lap * loss_laplacian

                # if (i + 1) % mod_iter == 0:
                #     # out_cpu = out.detach().cpu().numpy().squeeze()
                #     # plot_in_out_pointclouds(tgt_cpu, out_cpu)

                loss.sum().backward(retain_graph=True)

                return loss

        if i % 100 < 30:
            loss = opt1.step(closure)
            lr_schedule1.step(loss)
        else:
            loss = opt2.step(closure)
            lr_schedule2.step(loss)

        prev_loss = loss
        loss = opt1.step(closure)
        lr_schedule1.step(loss)

        out = layer((torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1),
                     torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))

        if (i + 1) % mod_iter == 0:
            save_name = "nurbs_fitting_" + str(i + 1) + ".png"
            visualize_nurbs_surface(out, inp_ctrl_pts, tgt_cpu, save_name)


        if loss.item() < 1e-6:
            print((time.time() - time1) / (i + 1))
            break
        # if abs(loss.item() - prev_loss.item()) < 0.00015:
        #     print("converged")
        #     break

        pbar.set_description("Loss %s: %s" % (i + 1, loss.item()))

    # print((time.time() - time1) / (num_epochs + 1))

    # sys.exit()
    # Assuming the optimization loop has finished
    # and you have access to the final tensors

    # Re-create the fixed clamping knot tensors
    p = net_params['p']
    q = net_params['q']

    knot_rep_p_0 = torch.zeros(1, p + 1).cuda()
    knot_rep_p_1 = torch.zeros(1, p).cuda()
    knot_rep_q_0 = torch.zeros(1, q + 1).cuda()
    knot_rep_q_1 = torch.zeros(1, q).cuda()


    full_knot_intervals_v = torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), dim=-1)
    final_knots_v = torch.cumsum(full_knot_intervals_v, dim=-1)
    full_knot_intervals_u = torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), dim=-1)
    final_knots_u = torch.cumsum(full_knot_intervals_u, dim=-1)

    # Normalize the knot vector to the [0, 1] range
    max_val = final_knots_v.max()
    if max_val > 0:
        final_knots_v_normalized = final_knots_v / max_val
    else:
        final_knots_v_normalized = final_knots_v
    # Normalize the knot vector to the [0, 1] range
    max_val = final_knots_u.max()
    if max_val > 0:
        final_knots_u_normalized = final_knots_u / max_val
    else:
        final_knots_u_normalized = final_knots_u


    final_knots_u_list = final_knots_u_normalized.squeeze().tolist()
    final_knots_v_list = final_knots_v_normalized.squeeze().tolist()
    return inp_ctrl_pts, loss, final_knots_u_list, final_knots_v_list


def visualize_nurbs_surface(out, inp_ctrl_pts, tgt_cpu, save_name=""):
    fig = plt.figure()
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    out_cpu = out.detach().cpu().numpy().squeeze()



    ax2 = fig.add_subplot(projection='3d')

    # Plot the predicted surface and control points
    surf2 = ax2.plot_wireframe(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green',
                               label='Predicted Surface')
    surf2 = ax2.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
                               linestyle='dashed', color='orange', label='Predicted Control Points')

    # Plot the target points
    ax2.scatter(tgt_cpu[:, 0], tgt_cpu[:, 1], tgt_cpu[:, 2], c='r', marker='o')

    # Set the camera view (azimuth, distance, elevation)
    # ax2.azim = 45
    # ax2.dist = 9.5
    # ax2.elev = 30
    ax2.azim = -130
    ax2.dist = 9.5
    ax2.elev = 45


    # Add axis labels
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')

    # Display ticks for all axes
    ax2.set_xticks(np.linspace(tgt_cpu[:, 0].min(), tgt_cpu[:, 0].max(), num=5))
    ax2.set_yticks(np.linspace(tgt_cpu[:, 1].min(), tgt_cpu[:, 1].max(), num=5))
    ax2.set_zticks(np.linspace(tgt_cpu[:, 2].min(), tgt_cpu[:, 2].max(), num=5))

    diffX = 0.2 * (tgt_cpu[:, 0].max() - tgt_cpu[:, 0].min())
    diffY = 0.2 * (tgt_cpu[:, 1].max() - tgt_cpu[:, 1].min())
    diffZ = 0.2 * (tgt_cpu[:, 2].max() - tgt_cpu[:, 2].min())

    ax2.set_xlim(tgt_cpu[:, 0].min() - diffX, tgt_cpu[:, 0].max() + diffX)
    ax2.set_ylim(tgt_cpu[:, 1].min() - diffY, tgt_cpu[:, 1].max() + diffY)
    ax2.set_zlim(tgt_cpu[:, 2].min() - diffZ, tgt_cpu[:, 2].max() + diffZ)

    # Customize the pane colors for better visibility
    ax2.xaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))  # Light gray background
    ax2.yaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))
    ax2.zaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))

    #set aspect to equal
    ax2.set_aspect('equal')


    # Remove grid for a cleaner look
    ax2.grid(True)

    # Show the plot with axis labels and ticks
    # plt.show()
    save_path = '/home/lizeth/Downloads/gif'
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')


def visualize_input_grid_points(inp_ctrl_pts, tgt_cpu):
    fig = plt.figure()

    ax2 = fig.add_subplot(projection='3d')

    surf2 = ax2.plot_wireframe(inp_ctrl_pts[:, :, 0], inp_ctrl_pts[:, :, 1], inp_ctrl_pts[:, :, 2],
                               linestyle='dashed', color='orange', label='Predicted Control Points')

    # Plot the target points
    ax2.scatter(tgt_cpu[:, 0], tgt_cpu[:, 1], tgt_cpu[:, 2], c='r', marker='o')

    # Set the camera view (azimuth, distance, elevation)
    ax2.azim = 45
    ax2.dist = 6.5
    ax2.elev = 30

    # Add axis labels
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')

    # Display ticks for all axes
    ax2.set_xticks(np.linspace(tgt_cpu[:, 0].min(), tgt_cpu[:, 0].max(), num=5))
    ax2.set_yticks(np.linspace(tgt_cpu[:, 1].min(), tgt_cpu[:, 1].max(), num=5))
    ax2.set_zticks(np.linspace(tgt_cpu[:, 2].min(), tgt_cpu[:, 2].max(), num=5))

    # Customize the pane colors for better visibility
    ax2.xaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))  # Light gray background
    ax2.yaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))
    ax2.zaxis.set_pane_color((0.9, 0.9, 0.9, 0.8))

    # Remove grid for a cleaner look
    ax2.grid(True)

    # Show the plot with axis labels and ticks
    plt.show()


def region_growing(k, max_iterations, points):
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)

    # Start the region growing process
    region = set([0])  # Start with the initial point index
    new_points = set([0])

    for iteration in range(max_iterations):
        if not new_points:
            break
        current_points = list(new_points)
        new_points = set()
        for point_idx in current_points:
            distances, indices = nbrs.kneighbors([points[point_idx]])
            for neighbor_idx in indices[0]:
                if neighbor_idx not in region:
                    region.add(neighbor_idx)
                    new_points.add(neighbor_idx)

    # Extract the region points
    region_points = points[list(region)]

    # Plotting the region for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(region_points[:, 0], region_points[:, 1], region_points[:, 2], c='r', marker='o')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', alpha=0.1)
    plt.show()

    return region_points

def compute_pca(tensor):
    # Center the data
    mean = torch.mean(tensor, dim=0)
    centered_data = tensor - mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered_data.T, centered_data) / (centered_data.size(0) - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return mean, sorted_eigenvectors

def region_growing_fitting(net_params, k, max_iterations, min_points, points):
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    n_ctrpts_u = net_params['n_ctrpts']
    n_ctrpts_v = net_params['n_ctrpts']

    # Start the region growing process
    region = set([0])  # Start with the initial point index
    new_points = set([0])
    new_grid = True

    for iteration in range(max_iterations):
        if not new_points:
            break
        current_points = list(new_points)
        new_points = set()
        if len(region) >= min_points and new_grid == True:
            verts = points[list(region)]
            target_vert = torch.tensor(verts).float().cuda()
            mean, pca_components = compute_pca(target_vert)
            grid_points = create_offset_grid(target_vert, pca_components, n_ctrpts_u, n_ctrpts_v)
            ctrl_points, loss = nurbs_fitting(net_params, grid_points, target_vert)
            new_grid = False

        for point_idx in current_points:
            distances, indices = nbrs.kneighbors([points[point_idx]])
            for neighbor_idx in indices[0]:
                if neighbor_idx not in region:
                    region.add(neighbor_idx)
                    new_points.add(neighbor_idx)
        if new_grid == False:
            verts = points[list(region)]
            target_vert = torch.tensor(verts).float().cuda()
            target_vert = torch.tensor(verts).float().cuda()
            mean, pca_components = compute_pca(target_vert)
            grid_points = create_offset_grid(target_vert, pca_components, n_ctrpts_u, n_ctrpts_v)

            ctrl_points, loss = nurbs_fitting(net_params, grid_points.squeeze(), target_vert)
            loss

    # Extract the region points
    region_points = points[list(region)]

    # Plotting the region for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(region_points[:, 0], region_points[:, 1], region_points[:, 2], c='r', marker='o')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', alpha=0.1)
    plt.show()

    return region_points


def visualize_pca_grid_points(target_vert, grid_points_global, projected_points):
    #reshape into  16 3
    grid_points_list= grid_points_global.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    target_vert_cpu = target_vert.detach().cpu().numpy().squeeze()
    projected_points_cpu = projected_points.detach().cpu().numpy().squeeze()
    grid_points_list_cpu = grid_points_list.detach().cpu().numpy().squeeze()
    ax.scatter(target_vert_cpu[:, 0], target_vert_cpu[:, 1], target_vert_cpu[:, 2], c='r', marker='o')
    zeros = np.zeros_like(projected_points_cpu[:, 0])
    ax.scatter(projected_points_cpu[:, 0], projected_points_cpu[:, 1], zeros, c='b', marker='o')
    ax.scatter(grid_points_list_cpu[:, 0], grid_points_list_cpu[:, 1], grid_points_list_cpu[:, 2], c='g', marker='o')
    plt.show()

def create_offset_grid(target_vert, eigenvectors, npoints_x, npoints_y):
    # Project the target points onto the first two principal components
    projected_points = torch.matmul(target_vert, eigenvectors[:, :2])

    # Compute the bounding box in the 2D plane
    min_x, _ = torch.min(projected_points[:, 0], dim=0)
    max_x, _ = torch.max(projected_points[:, 0], dim=0)
    min_y, _ = torch.min(projected_points[:, 1], dim=0)
    max_y, _ = torch.max(projected_points[:, 1], dim=0)

    # Create a grid within the bounding box
    x = torch.linspace(min_x, max_x, npoints_x, device=target_vert.device)
    y = torch.linspace(min_y, max_y, npoints_y, device=target_vert.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    zz = torch.zeros_like(xx, device=target_vert.device)
    grid_points_local = torch.stack([xx, yy, zz], dim=-1)

    #transpose x and y
    grid_points_local = grid_points_local.permute(1, 0, 2)

    # Transform grid points to the original coordinate system
    grid_points_global = torch.einsum('ijk,kl->ijl', grid_points_local, eigenvectors[:, :3].T)

    visualize_pca_grid_points(target_vert, grid_points_global, projected_points)
    # grid_points_global = grid_points_global.T.reshape(3, npoints_x, npoints_y)
    return grid_points_global




def map_distances_to_colors(distances):
    """
    Map distances to a color scale.

    Parameters:
    - distances: A numpy array of shape (n,) containing the distances.

    Returns:
    - colors: A numpy array of shape (n, 3) containing the RGB colors.
    """
    # Normalize distances between 0 and 1
    norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    # Use a colormap to map normalized distances to colors
    colormap = plt.cm.viridis
    colors = colormap(norm_distances)
    return colors


def visualize_point_cloud_with_colors(points, colors, distances, colormap_name='viridis'):
    """
    Visualize a point cloud with colors using matplotlib, including a colorbar for distances.

    Parameters:
    - points: A numpy array of shape (n, 3) containing the points (x, y, z).
    - colors: A numpy array of shape (n, 3) containing the RGB colors.
    - distances: A numpy array of shape (n,) containing the distances of each point to the mesh.
    - colormap_name: The name of the colormap to use.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, marker='o')

    # Create a colorbar
    colormap = plt.get_cmap(colormap_name)
    norm = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical')
    cbar.set_label('Distance to Surface Mesh')

    plt.show()


def get_random_color_from_colormap(colormap_name='viridis'):
    """
    Get a random color from a specified colormap.

    Parameters:
    - colormap_name: The name of the colormap to use (default is 'viridis').

    Returns:
    - color: A tuple representing the chosen color in RGB format (0-255).
    """
    # Get the colormap from matplotlib
    colormap = plt.get_cmap(colormap_name)

    # Generate a random value between 0 and 1
    random_value = np.random.rand()

    # Get the color corresponding to the random value from the colormap
    color = colormap(random_value)

    # Convert the color from [0, 1] range to [0, 255] range and to RGB format
    color_rgb = (np.array(color[:3]) * 255).astype(int)

    return tuple(color_rgb)

def extract_numbers_from_line(line):
    import re
    # Define a regular expression pattern to find all numbers in the line
    pattern = r'\d+\.?\d*'  # This will match integers and decimal numbers

    # Find all matches of the pattern in the line
    matches = re.findall(pattern, line)

    # Convert matches to float or int
    numbers = [float(match) if '.' in match else int(match) for match in matches]

    return numbers
def load_primitives_from_vg(primitives_file):
    points = []
    normals = []
    groups = []
    planes = []
    with open(primitives_file, 'r') as file:
        # Read the file line by line
        with open(primitives_file, 'r') as file:
            # Extract numbers from the current line
            number_line = file.readline()
            line_numbers = extract_numbers_from_line(number_line)
            n_poins = line_numbers[0]
            #read the n_points
            for i in range(n_poins):
                line = file.readline()
                [x, y, z] = line.split()
                points.append([float(x), float(y), float(z)])

            number_line = file.readline()

            for i in range(n_poins):
                line = file.readline()

            number_line = file.readline()

            for i in range(n_poins):
                line = file.readline()
                [x, y, z] = line.split()
                normals.append([float(x), float(y), float(z)])

            number_line = file.readline()
            line_numbers = extract_numbers_from_line(number_line)
            n_groups = line_numbers[0]

            for i in range(n_groups):
                number_line = file.readline()
                number_line = file.readline()
                number_line = file.readline()
                [t, a, b, c, d] = number_line.split()
                h_list = [float(a), float(b), float(c), float(d)]
                planes.append(h_list)
                number_line = file.readline()
                number_line = file.readline()
                number_line = file.readline()
                line_numbers = extract_numbers_from_line(number_line)
                line = file.readline()
                str_numbers = line.split()
                int_list = [int(num) for num in str_numbers]
                groups.append(int_list)
                number_line = file.readline()

    return points, normals, groups, planes


def clean_directory(directory, extensions=('.png', '.gif')):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Loop over each specified file extension
    for ext in extensions:
        # Find all files with the given extension
        files = glob.glob(os.path.join(directory, f'*{ext}'))

        # Remove each file found
        for file in files:
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except Exception as e:
                print(f"Failed to remove {file}: {e}")

    print("Directory cleanup completed.")


def create_gif(directory, output_path, duration=500):
    # Collect all PNG files in the specified directory
    image_files = sorted(glob.glob(os.path.join(directory, 'merge_*.png')))  # Sort to maintain sequence

    if not image_files:
        print(f"No images found in the directory: {directory}")
        return

    # Open images and create a GIF
    images = [Image.open(image_file) for image_file in image_files]
    images[0].save(directory + output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    print(f"GIF saved to {output_path}")

def read_adjacency_list(filename):
    adjacency_list = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            num_adjacent = int(parts[0])
            if num_adjacent > 0:
                indices = list(map(int, parts[1:1 + num_adjacent]))
            else:
                indices = []
            adjacency_list.append(indices)

    return adjacency_list


def load_mesh_and_point_cloud(mesh_file, pc_file):
    """
    Load a mesh and point cloud using trimesh.
    """
    # Load the mesh using trimesh
    mesh = trimesh.load(mesh_file)

    # Load the point cloud as a Trimesh point cloud
    point_cloud = trimesh.load(pc_file)

    # Ensure the point cloud contains only vertices (no faces)
    if point_cloud.is_empty:
        raise ValueError(f"Point cloud {pc_file} is empty!")

    return mesh, point_cloud


def trim_mesh(mesh, pc_points, distance_threshold=0.05):
    """
       Trim triangles from a mesh based on the point-to-mesh distance from a point cloud.
       Triangles that are too far from the point cloud are removed.
       Computes the sum of inlier distances, number of inliers, and number of points per patch.

       Parameters:
       - mesh: trimesh.Trimesh object representing the 3D mesh.
       - point_cloud: trimesh.PointCloud object representing the point cloud.
       - distance_threshold: float, the distance threshold for determining inliers.

       Returns:
       - trimmed_mesh: trimesh.Trimesh object with triangles trimmed based on the point cloud proximity.
       - sum_inlier_distances: float, the sum of distances for inlier points.
       - number_of_inliers: int, the count of inliers in the point cloud.
       - points_per_patch: list of ints, where each entry represents the number of inliers associated with each triangle.
       """
    # Extract vertices and faces (triangles) from the mesh
    mesh_vertices = mesh.vertices
    mesh_triangles = mesh.faces  # In trimesh, 'faces' refers to triangles

    # Handle vertex or face colors if available
    has_vertex_colors = hasattr(mesh, 'visual') and mesh.visual.kind == 'vertex'
    has_face_colors = hasattr(mesh, 'visual') and mesh.visual.kind == 'face'

    vertex_colors = mesh.visual.vertex_colors if has_vertex_colors else None
    face_colors = mesh.visual.face_colors if has_face_colors else None

    # Get the closest points on the mesh to the points in the point cloud using trimesh.proximity.closest_point
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, pc_points)

    # _, dist1, _ = mesh.nearest.on_surface(pc_points)

    # Compute inliers based on the distance threshold
    inlier_mask = distances < distance_threshold
    inlier_distances = distances[inlier_mask]

    # Compute the sum of inlier distances and the number of inliers
    sum_inlier_distances = np.sum(inlier_distances)
    number_of_inliers = np.sum(inlier_mask)

    # Build a KD-Tree from the closest points for efficient nearest-neighbor search
    kdtree = cKDTree(closest_points)

    # List to store triangles that are kept after trimming
    trimmed_triangles = []
    trimmed_face_colors = []  # To store face colors if face colors are used
    trimmed_vertex_colors = np.zeros_like(mesh_vertices) if has_vertex_colors else None

    # List to store the number of inliers per triangle
    points_per_patch = []

    # Check each triangle in the mesh
    for i, triangle_indices in enumerate(mesh_triangles):
        # Get the vertices of the triangle
        triangle_vertices = mesh_vertices[triangle_indices]

        # Find the distance from each triangle vertex to the nearest point in the point cloud
        distances_to_triangle, _ = kdtree.query(triangle_vertices)

        # Count the number of inliers associated with this triangle
        inliers_per_triangle = np.sum(distances_to_triangle < distance_threshold)
        points_per_patch.append(inliers_per_triangle)

        # If the minimum distance between any triangle vertex and the point cloud is below the threshold, keep the triangle
        if np.any(distances_to_triangle < distance_threshold):
            trimmed_triangles.append(triangle_indices)
            if has_face_colors:
                trimmed_face_colors.append(face_colors[i])

            # Propagate vertex colors if available
            if has_vertex_colors:
                trimmed_vertex_colors[triangle_indices] = vertex_colors[triangle_indices]

    # Create a new trimesh object with the trimmed triangles
    trimmed_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=trimmed_triangles)

    # If face colors are used, assign them to the new mesh
    if has_face_colors:
        trimmed_mesh.visual.face_colors = np.array(trimmed_face_colors)

    # If vertex colors are used, assign them to the new mesh
    if has_vertex_colors:
        trimmed_mesh.visual.vertex_colors = trimmed_vertex_colors

    return trimmed_mesh, sum_inlier_distances, number_of_inliers, points_per_patch


def process_trim_patches(mesh_files, pc_files, mesh_dir, points_dir, distance_threshold=0.05):
    """
    Process a list of mesh and point cloud files, trimming triangles that are not near points,
    and compute fidelity, completeness, and simplicity after processing all meshes.

    Parameters:
    - mesh_files: list of file paths for mesh files (mesh patches).
    - pc_files: list of file paths for point cloud files associated with each patch.
    - distance_threshold: float, distance threshold to consider points as inliers.

    Returns:
    - final_metrics: A dictionary containing final metrics (fidelity, completeness, simplicity).
    """
    # Initialize accumulators
    total_sum_inlier_distances = 0.0
    total_number_of_inliers = 0
    total_points = 0
    simplicity = len(mesh_files)  # Simplicity is the number of mesh patches

    for mesh_file, pc_file in zip(mesh_files, pc_files):
        # Load the mesh and point cloud using trimesh
        mesh_save_dir = mesh_dir + '/trimmed/'
        metrics_save_dir = mesh_dir + '/metrics.txt'

        if not os.path.exists(mesh_save_dir):
            os.makedirs(mesh_save_dir)

        trimmed_mesh_file = os.path.join(mesh_save_dir, mesh_file)



        mesh = trimesh.load(mesh_dir + mesh_file)
        point_cloud = trimesh.load(points_dir + pc_file)

        # Get total number of points in the current point cloud
        num_points_in_pc = len(point_cloud.vertices)
        total_points += num_points_in_pc

        # Trim the mesh based on proximity to the point cloud and gather metrics
        trimmed_mesh, sum_inlier_distances, number_of_inliers, _ = trim_mesh(mesh, point_cloud,distance_threshold)

        # Accumulate inlier distances and inlier counts
        total_sum_inlier_distances += sum_inlier_distances
        total_number_of_inliers += number_of_inliers

        trimmed_mesh.export(trimmed_mesh_file)
        print(f"Saved trimmed mesh to {trimmed_mesh_file}")

    # After processing all patches, compute final metrics:

    # Compute fidelity as the mean inlier distance (only if there are inliers)
    fidelity = total_sum_inlier_distances / total_number_of_inliers if total_number_of_inliers > 0 else 0

    # Compute completeness as 1 - (total number of inliers / total number of points)
    completeness =  (total_number_of_inliers / total_points) if total_points > 0 else 0

    # Simplicity is the number of mesh patches processed (already stored as 'simplicity')

    # Print final results
    print("\nFinal Results After Processing All Mesh Patches:")
    print(f"  Fidelity: {fidelity}")
    print(f"  Completeness: {completeness}")
    print(f"  Simplicity (number of patches): {simplicity}")
    print(f"  Total Sum of Inlier Distances: {total_sum_inlier_distances}")
    print(f"  Total Number of Inliers: {total_number_of_inliers}")
    print(f"  Total Number of Points: {total_points}")

    # Return final metrics as a dictionary
    final_metrics = {
        "fidelity": fidelity,
        "completeness": completeness,
        "simplicity": simplicity,
        "sum_inlier_distances": total_sum_inlier_distances,
        "number_of_inliers": total_number_of_inliers,
        "total_points": total_points
    }
    # save the metrics like this format
    with open(metrics_save_dir , 'w') as file:
        for key, value in final_metrics.items():
            file.write(f"{key}: {value}\n")

    return final_metrics


def get_files_in_directory(directory):
    file_list = []

    # Loop through the directory
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)

        # Check if it's a file, not a directory
        if os.path.isfile(full_path):
            file_list.append(full_path)

    return file_list


def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def visualize_normals(v1, v2):
# visualize the normals and the points and show the angle
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up the origin for both vectors
    origin = np.array([0, 0, 0])

    # Plot the vectors starting from the origin
    ax.quiver(*origin, *v1, color='r', label=f'Vector 1 (Value: {angle})')
    ax.quiver(*origin, *v2, color='b', label='Vector 2')

    # Set plot limits for better visualization
    ax.set_xlim([min(0, v1[0], v2[0]), max(0, v1[0], v2[0])])
    ax.set_ylim([min(0, v1[1], v2[1]), max(0, v1[1], v2[1])])
    ax.set_zlim([min(0, v1[2], v2[2]), max(0, v1[2], v2[2])])

    # Set labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Add legend to distinguish between the two vectors
    ax.legend()

    # Show the plot
    plt.show()

def read_metrics(metrics_file):
    """
    Read the metrics from a given file and return them as a dictionary.
    """
    metrics = {}
    with open(metrics_file, 'r') as file:
        for line in file:
            if line == '\n':
                continue
            key, value = line.strip().split()
            metrics[key] = value
    return metrics

def visualize_points(original_points, transformed_points, text=None):
    #add a legend to the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c='r', marker='o')
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='b', marker='o')

    # add a legend to the plot
    ax.legend(['points', 'grid points'], loc='upper right')
    ax.set_title(text)
    ax.set_aspect('equal')
    plt.show()


def clean_merged_folders(path_save, merged_surface_str):

    merged_surface_str = path_save + merged_surface_str
    merged_surface_color_str = merged_surface_str.replace('merged_surface', 'merged_surface_color')
    merged_points_str = merged_surface_str.replace('merged_surface', 'merged_surface_points')

    if os.path.exists(merged_surface_str):
        clean_directory(merged_surface_str)
    if os.path.exists(merged_surface_color_str):
        clean_directory(merged_surface_color_str)
    if os.path.exists(merged_points_str):
        clean_directory(merged_points_str)


def plot_planes_points(plane, points):
    """
    Plot two planes, their normal vectors, and points on the planes in 3D.

    Parameters:
    - plane1: Coefficients [a, b, c, d] of the first plane equation.
    - plane2: Coefficients [a, b, c, d] of the second plane equation.
    - points1: Array of points on the first plane.
    - points2: Array of points on the second plane.
    """

    def plot_plane(ax, xx, yy, zz, color, alpha=0.5):
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, rstride=100, cstride=100)

    def plot_normal_vector(ax, plane, centroid, color):
        a, b, c, d = plane
        normal_vector = np.array([a, b, c])
        ax.quiver(centroid[0], centroid[1], centroid[2], normal_vector[0], normal_vector[1], normal_vector[2],
                  color=color, length=0.1, arrow_length_ratio=0.1)

    def scale_plane_to_points(plane, points):
        """
        Scale the plane to fit the bounding box of the given points.

        Parameters:
        - plane: Coefficients [a, b, c, d] of the plane equation.
        - points: Array of points to fit the plane.

        Returns:
        - xx: Grid of x values.
        - yy: Grid of y values.
        - zz: Corresponding z values on the plane.
        """
        a, b, c, d = plane

        points = np.array(points)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # Define the grid limits based on the bounding box of the points
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

        # Compute the corresponding z values on the plane
        zz = (-a * xx - b * yy - d) / c

        return xx, yy, zz

    def compute_centroid(points):
        """
        Compute the centroid of a set of points.

        Parameters:
        - points: Array of points.

        Returns:
        - centroid: Coordinates of the centroid.
        """
        points = np.array(points)
        centroid = points.mean(axis=0)
        return centroid

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scale and plot the planes
    xx1, yy1, zz1 = scale_plane_to_points(plane, points)

    plot_plane(ax, xx1, yy1, zz1, color='blue', alpha=0.5)

    # Compute centroids
    centroid1 = compute_centroid(points)


    # Plot the normal vectors at the centroid of the points
    plot_normal_vector(ax, plane, centroid1, color='blue')


    # Plot the points on each plane
    points1 = np.array(points)

    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='blue', marker='o', label='Points on Plane 1')


    # #plot points normals
    # ax.quiver(points2[0], points2[1], points2[2], normals2[0], normals2[1], normals2[2],
    #           color='blue', length=0.2, arrow_length_ratio=0.1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Add a legend
    ax.legend()
    ax.set_aspect('equal')

    # Show the plot
    plt.show()

def visualize_projected_grid_points(points, grid_points, projected_points):
    #reshape into  16 3
    grid_points_list= grid_points.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c='g', marker='o')
    ax.scatter(grid_points_list[:, 0], grid_points_list[:, 1], grid_points_list[:, 2], c='r', marker='o')
    plt.show()


def plot_plane(plane_params, ax, colorp, points):
    a, b, c, d = plane_params

    # Determine the range of the input points to define the grid size
    min_x, max_x = min(points[:, 0]), max(points[:, 0])
    min_y, max_y = min(points[:, 1]), max(points[:, 1])
    min_z, max_z = min(points[:, 2]), max(points[:, 2])

    # Create a grid for the plane based on the points' range
    xx, yy = np.meshgrid(
        np.linspace(min_x, max_x, num=100),  # More samples for smoother plane
        np.linspace(min_y, max_y, num=100)
    )

    zz = (-d - a * xx - b * yy) / c

    # Adjust z-limits based on points to ensure the plane is within view
    ax.set_zlim(min(min_z, zz.min()), max(max_z, zz.max()))

    # Plot the plane
    ax.plot_surface(xx, yy, zz, color=colorp, alpha=0.3)

def plot_meshes_and_points(mesh1, mesh2, plane1, plane2, points1, points2, id1, id2, grid_points_3d, updated_points, plane_params, normal, merged_points):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    len_vec = 0.05

    # Plot mesh1
    mesh1_faces = mesh1.vertices[mesh1.faces]
    ax.add_collection3d(Poly3DCollection(mesh1_faces, facecolors='cyan', linewidths=0.1, edgecolors='r', alpha=0.5))

    # Plot mesh2
    mesh2_faces = mesh2.vertices[mesh2.faces]
    ax.add_collection3d(Poly3DCollection(mesh2_faces, facecolors='magenta', linewidths=0.1, edgecolors='b', alpha=0.5))

    # Plot original grid points
    ax.scatter(grid_points_3d[:, :, 0], grid_points_3d[:, :, 1], grid_points_3d[:, :, 2], color='blue', label='Original Points')

    # Plot updated points
    ax.scatter(updated_points[:, :,  0], updated_points[:,:, 1], updated_points[:, :, 2], color='red', label='Updated Points')

    plot_plane(plane_params, ax, 'yellow', merged_points)
    plot_plane(plane1, ax, 'cyan', merged_points)
    plot_plane(plane2, ax, 'magenta', merged_points)

    # Plot the normal
    centroid = np.mean(grid_points_3d.reshape(-1, 3), axis=0)
    normal_start = centroid
    normal_end = centroid + normal
    ax.quiver(*normal_start, *(normal_end - normal_start), color='black', length=len_vec, normalize=True, label='Normal avg plane')
    #plot mesh 1 normal
    centroid = np.mean(mesh1.vertices, axis=0)
    normal_start = centroid
    normal_end = centroid + plane1[:3]
    ax.quiver(*normal_start, *(normal_end - normal_start), color='cyan', length=len_vec, normalize=True, label='Normal patch1 plane')
    #plot mesh2 normal
    centroid = np.mean(mesh2.vertices.reshape(-1, 3), axis=0)
    normal_start = centroid
    normal_end = centroid + plane2[:3]
    ax.quiver(*normal_start, *(normal_end - normal_start), color='magenta', length=len_vec, normalize=True, label='Normal patch2 plane')



    # Plot the points on each plane
    points1 = np.array(points1)
    points2 = np.array(points2)
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='cyan', marker='o', label= id1)
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='magenta', marker='o', label= id2)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    min_x, max_x = min(merged_points[:, 0]), max(merged_points[:, 0])
    min_y, max_y = min(merged_points[:, 1]), max(merged_points[:, 1])
    min_z, max_z = min(merged_points[:, 2]), max(merged_points[:, 2])

    ax.set_zlim(min_z, max_z)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.set_aspect('equal', adjustable='box')

    ax.legend()
    plt.show()

def compute_centroid(points):
    """
    Compute the centroid of a set of points.

    Parameters:
    - points: Array of points.

    Returns:
    - centroid: Coordinates of the centroid.
    """
    points = np.array(points)
    centroid = points.mean(axis=0)
    return centroid

def scale_plane_to_points(plane, points):
    """
    Scale the plane to fit the bounding box of the given points.

    Parameters:
    - plane: Coefficients [a, b, c, d] of the plane equation.
    - points: Array of points to fit the plane.

    Returns:
    - xx: Grid of x values.
    - yy: Grid of y values.
    - zz: Corresponding z values on the plane.
    """
    a, b, c, d = plane

    points = np.array(points)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    # Define the grid limits based on the bounding box of the points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    # Compute the corresponding z values on the plane
    zz = (-a * xx - b * yy - d) / c

    return xx, yy, zz


def plot_planes_and_normals_with_points(plane1, plane2, points1, points2, additonal_points = None):
    """
    Plot two planes, their normal vectors, and points on the planes in 3D.

    Parameters:
    - plane1: Coefficients [a, b, c, d] of the first plane equation.
    - plane2: Coefficients [a, b, c, d] of the second plane equation.
    - points1: Array of points on the first plane.
    - points2: Array of points on the second plane.
    """

    def plot_plane(ax, xx, yy, zz, color, alpha=0.5):
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, rstride=100, cstride=100)

    def plot_normal_vector(ax, plane, centroid, color):
        a, b, c, d = plane
        normal_vector = np.array([a, b, c])
        ax.quiver(centroid[0], centroid[1], centroid[2], normal_vector[0], normal_vector[1], normal_vector[2],
                  color=color, length=0.1, arrow_length_ratio=0.1)

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scale and plot the planes
    xx1, yy1, zz1 = scale_plane_to_points(plane1, points1)
    xx2, yy2, zz2 = scale_plane_to_points(plane2, points2)

    plot_plane(ax, xx1, yy1, zz1, color='blue', alpha=0.5)
    plot_plane(ax, xx2, yy2, zz2, color='green', alpha=0.5)

    # Compute centroids
    centroid1 = compute_centroid(points1)
    centroid2 = compute_centroid(points2)

    # Plot the normal vectors at the centroid of the points
    plot_normal_vector(ax, plane1, centroid1, color='blue')
    plot_normal_vector(ax, plane2, centroid2, color='green')

    # Plot the points on each plane
    points1 = np.array(points1)
    points2 = np.array(points2)
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='blue', marker='o', label='Points on Plane 1')
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='green', marker='o', label='Points on Plane 2')

    if additonal_points is not None:
        ax.scatter(additonal_points[:, 0], additonal_points[:, 1], additonal_points[:, 2], color='red', marker='o', label='Additional Points')

    # #plot points normals
    # ax.quiver(points2[0], points2[1], points2[2], normals2[0], normals2[1], normals2[2],
    #           color='blue', length=0.2, arrow_length_ratio=0.1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def filter_and_remove_points(permanent_set, additional_set, threshold):
    """
    Filters points from the additional set that are within the threshold distance
    from any point in the permanent set and removes them from the additional set.

    Parameters:
        permanent_set (numpy.ndarray): Array of shape (n, d), where `n` is the number of points
                                       and `d` is the dimensionality of each point.
        additional_set (numpy.ndarray): Array of shape (m, d), where `m` is the number of additional points.
        threshold (float): Distance threshold for filtering points.

    Returns:
        tuple: (filtered_points, remaining_additional_set)
               filtered_points (numpy.ndarray) - Points from additional set within the threshold.
               remaining_additional_set (numpy.ndarray) - Points from additional set not within the threshold.
    """
    if additional_set.size == 0:
        return np.array([]), np.array([])

    filtered_points = []
    remaining_points = []

    for point in additional_set:
        distances = np.linalg.norm(permanent_set - point,
                                   axis=1)  # Calculate distances to all points in the permanent set
        if np.min(distances) <= threshold:
            filtered_points.append(point)  # Add to filtered if within threshold
        else:
            remaining_points.append(point)  # Add to remaining if not within threshold

    return np.array(filtered_points), np.array(remaining_points)


def load_meshes_points_from_folder(mesh_folder, points_folder, prefix='_surfc'):

    meshes = []
    point_clouds = []
    for shape_name in os.listdir(mesh_folder):
        points_name = shape_name.replace(prefix, '_points')
        mesh = trimesh.load_mesh(mesh_folder + shape_name)
        pointcloud = trimesh.load_mesh(points_folder + points_name)
        points = np.asarray(pointcloud.vertices)
        meshes.append(mesh)
        point_clouds.append(pointcloud)
    return meshes, point_clouds


def assign_points_to_patches(input_points, surface_points, patch_labels, epsilon):
    """
    Assigns each point in input_points to the closest patch based on surface points and their labels.

    :param input_points: Array of points to classify, shape (N, 3)
    :param surface_points: Array of surface points, shape (M, 3)
    :param patch_labels: Array of patch labels corresponding to surface points, shape (M,)
    :param epsilon: Maximum allowable distance to consider a point part of a patch
    :return: Array of labels for each point in input_points, shape (N,)
             Label is -1 for outliers.
    """
    # Build KDTree for fast nearest neighbor search
    kdtree = KDTree(surface_points)

    # Query the nearest surface point for each input point
    distances, indices = kdtree.query(input_points)

    # Initialize labels array
    labels = np.full(input_points.shape[0], -1, dtype=int)

    # Assign labels based on distance and patch labels
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist < epsilon:
            labels[i] = patch_labels[idx]
        else:
            labels[i] = -1  # Mark as outlier

    return labels


def visualize_points_with_labels(input_points, labels):
    """
    Visualizes the input points in 3D, colored by their labels.

    :param input_points: Array of points, shape (N, 3)
    :param labels: Array of labels for each point, shape (N,)
    """
    unique_labels = np.unique(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assign a color for each label
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_mask = labels == label
        if label == -1:
            ax.scatter(input_points[label_mask, 0], input_points[label_mask, 1], input_points[label_mask, 2],
                       c=["black"], label="Outliers", s=10)
        else:
            ax.scatter(input_points[label_mask, 0], input_points[label_mask, 1], input_points[label_mask, 2],
                       c=[color], label=f"Patch {label}", s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def visualize_point_sets(points_a, points_b, outliers):
    """
    Visualizes two sets of 3D points with different colors and marks outliers in black.

    :param points_a: Array of points in set A, shape (N_a, 3)
    :param points_b: Array of points in set B, shape (N_b, 3)
    :param outlier_indexes: Array of indices corresponding to outlier points
    """
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot set A points in blue
    ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], c='green', label='Set A', s=10)

    # Plot set B points in green
    ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], c='blue', label='Set B', s=10)

    # Plot outliers in black
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='black', label='Outliers', s=30)

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def delete_files_with_extensions(directory, extensions):
    """
    Deletes files with specified extensions in a given directory (non-recursively).

    :param directory: Path to the directory containing files.
    :param extensions: List of file extensions to delete (e.g., ['.aux', '.log', '.pdf']).
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    deleted_files = 0  # Counter for deleted files
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file (not a directory) and its extension matches
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext in extensions:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"Deleted {deleted_files} files with extensions {extensions} from {directory}.")

def copy_shape_file(shape_path, path_output, shape_name, method_name):
    output_shape_file = os.path.join(path_output, shape_name,  shape_name + '_' + method_name + '.ply')
    #create the output directory if it does not exist
    if not os.path.exists(os.path.join(path_output, shape_name)):
        os.makedirs(os.path.join(path_output, shape_name))
    # Copy the shape file to the output directory
    shutil.copyfile(shape_path, output_shape_file)

color_silo = []
def return_random_color(n_colors):
    # Generate a random RGB color
    global color_silo
    if len(color_silo) == 0:
        # Fill the color silo with all viridis colors
        cmap = plt.get_cmap('viridis')
        color_silo = [cmap(i) for i in np.linspace(0.1, 0.9, 20)]


    random_color = color_silo.pop()

    # random_color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)

    # Convert to RGBA (add alpha channel)
    # color_rgba = np.hstack([random_color, [[255]]])  # Adding full opacity
    #
    # return np.tile(color_rgba, (n_colors, 1))
    color_rgb = (np.array(random_color[:3]) * 255).astype(int)

    return tuple(color_rgb)

import numpy as np 
import os, sys
import trimesh
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def create_uv_mesh(x_res, y_res, debug=False):
    x = np.linspace(1, -1, x_res)
    y = np.linspace(1, -1, y_res)

    # exchange x,y to make everything consistent:
    # x is the first coordinate, y is the second!
    xv, yv = np.meshgrid(y, x)
    uv = np.stack((xv, yv), axis=-1)
    vertices = uv.reshape(-1, 2)
    
    tri = Delaunay(vertices)
    faces = tri.simplices
    vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=-1)

    if debug:
        # x in plt is vertical
        # y in plt is horizontal
        plt.figure()
        plt.triplot(vertices[:,0], vertices[:,1], faces)
        plt.plot(vertices[:,0], vertices[:,1], 'o', markersize=2)
        plt.savefig('../tmp/tri.png')

    return vertices, faces

def select_boundary(mesh, return_groups=False):
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    unique_edges = mesh.edges[groups]
    idx_boundary_v = np.unique(unique_edges.flatten())
    if return_groups:
        return idx_boundary_v, groups
    else:
        return idx_boundary_v

def reorder_vertices_faces(vertices, faces_new):
    faces_new_flatten = faces_new.reshape(-1)
    mask = np.zeros((len(vertices))).astype(bool)
    mask[faces_new_flatten] = True
    vertices_reorder = vertices[mask]

    re_id = np.zeros((len(vertices))).astype(int) - 1
    re_id[mask] = np.arange(len(vertices_reorder))
    faces_reorder = re_id[faces_new_flatten].reshape(-1, 3)

    return vertices_reorder, faces_reorder


def read_mesh_from_sdf(vertices, faces, sdf, edges, thresh=0):
    batch_size = len(sdf)
    
    edge_sdf = sdf[:,edges.reshape(-1)].reshape(batch_size, -1, 2) # (n, e, 2)
    
    edge_sdf_count = (edge_sdf<thresh).float().sum(dim=-1) == 1 # (n, e)
    idx_b, idx_e = torch.where(edge_sdf_count)
    
    v0 = vertices[idx_b, edges[idx_e, 0]]
    v1 = vertices[idx_b, edges[idx_e, 1]]
    v0_sdf = edge_sdf[idx_b, idx_e, 0][:, None]
    v1_sdf = edge_sdf[idx_b, idx_e, 1][:, None]

    v_border = ((v0*(v1_sdf-thresh)) + v1*(thresh-v0_sdf))/(v1_sdf - v0_sdf) # (n, e, 3)

    _, idx_v = torch.where(edge_sdf[idx_b,idx_e] >= thresh)
    idx_v = edges[idx_e, idx_v]

    vertices[idx_b, idx_v] = v_border

    tri_sdf = sdf[:, faces.reshape(-1)].reshape(batch_size, -1, 3)
    flag_in = (tri_sdf<thresh).sum(dim=-1)

    faces_list = []
    for b in range(batch_size):
        tri_in = faces[flag_in[b]>0]
        faces_list.append(tri_in)

    return vertices, faces_list


def triangulation_2D(boundary_f, boundary_b, idx_boundary_v_f, idx_boundary_v_b, idx_offset, xy='x', reverse=False):
    if xy=='x':
        axis = 0
    else:
        axis = 1
    len_f = len(boundary_f)
    len_b = len(boundary_b)

    idx_f = np.argsort(boundary_f[:, axis])[::-1] # increasing
    idx_b = np.argsort(boundary_b[:, axis])[::-1]
    idx_f = idx_boundary_v_f[idx_f]
    idx_b = idx_boundary_v_b[idx_b] + idx_offset

    len_cut = min(len_f, len_b)

    idx_0 = idx_f[:len_cut-1]
    idx_1 = idx_f[1:len_cut]
    idx_2 = idx_b[:len_cut-1]
    face1 = np.stack((idx_0, idx_1, idx_2), axis=-1)

    idx_0 = idx_b[1:len_cut]
    idx_1 = idx_b[:len_cut-1]
    idx_2 = idx_f[1:len_cut]
    face2 = np.stack((idx_0, idx_1, idx_2), axis=-1)

    faces = np.concatenate((face1, face2), axis=0)

    if len_f<len_b:
        idx_0 = idx_b[len_cut:]
        idx_1 = idx_b[len_cut-1:-1] 
        idx_2 = np.repeat(idx_f[-1], len(idx_0))
        face3 = np.stack((idx_0, idx_1, idx_2), axis=-1)
        faces = np.concatenate((faces, face3), axis=0)
    elif len_f>len_b:
        idx_0 = idx_f[len_cut-1:-1]
        idx_1 = idx_f[len_cut:]
        idx_2 = np.repeat(idx_b[-1], len(idx_0))
        face3 = np.stack((idx_0, idx_1, idx_2), axis=-1)
        faces = np.concatenate((faces, face3), axis=0)

    if reverse:
        faces = faces[:, [0,2,1]]
    return faces

def get_body_idx(mesh_pattern, idx_boundary_v, label, front=True):
    idx_boundary_body_bottom = idx_boundary_v[label==0]
    if front:
        idx_boundary_body_left = idx_boundary_v[label==1]
        idx_boundary_body_right = idx_boundary_v[label==10]
    else:
        idx_boundary_body_left = idx_boundary_v[label==1]
        idx_boundary_body_right = idx_boundary_v[label==9]

    v_bottom = mesh_pattern.vertices[idx_boundary_body_bottom]
    v_left = mesh_pattern.vertices[idx_boundary_body_left]
    v_right = mesh_pattern.vertices[idx_boundary_body_right]

    idx_bottom_r2l = np.argsort(v_bottom[:, 0])
    if v_left[:, 1].min() - v_bottom[idx_bottom_r2l[-1], 1] > 1e-6:
        idx_boundary_body_left = np.append(idx_boundary_body_left, idx_boundary_body_bottom[idx_bottom_r2l[-1]])

    if v_right[:, 1].min() - v_bottom[idx_bottom_r2l[0], 1] > 1e-6:
        idx_boundary_body_right = np.append(idx_boundary_body_right, idx_boundary_body_bottom[idx_bottom_r2l[0]])

    idx_highest_left = idx_boundary_body_left[np.argmax(v_left[:, 1])]
    idx_highest_right = idx_boundary_body_right[np.argmax(v_right[:, 1])]
    return idx_boundary_body_left, idx_boundary_body_right, idx_highest_left, idx_highest_right

def get_sleeve_idx(mesh_pattern, idx_boundary_v, label, front=True):
    if front:
        idx_boundary_cuff_left = idx_boundary_v[label==3]
        idx_boundary_cuff_right = idx_boundary_v[label==8]

        idx_boundary_sleeve_left_up = idx_boundary_v[label==4]
        idx_boundary_sleeve_right_up = idx_boundary_v[label==7]
        idx_boundary_sleeve_left_bottom = idx_boundary_v[label==2]
        idx_boundary_sleeve_right_bottom = idx_boundary_v[label==9]
    else:
        idx_boundary_cuff_left = idx_boundary_v[label==3]
        idx_boundary_cuff_right = idx_boundary_v[label==7]

        idx_boundary_sleeve_left_up = idx_boundary_v[label==4]
        idx_boundary_sleeve_right_up = idx_boundary_v[label==6]
        idx_boundary_sleeve_left_bottom = idx_boundary_v[label==2]
        idx_boundary_sleeve_right_bottom = idx_boundary_v[label==8]

    v_cuff_left = mesh_pattern.vertices[idx_boundary_cuff_left]
    v_cuff_right = mesh_pattern.vertices[idx_boundary_cuff_right]

    v_left_up = mesh_pattern.vertices[idx_boundary_sleeve_left_up]
    v_right_up = mesh_pattern.vertices[idx_boundary_sleeve_right_up]
    v_left_bottom = mesh_pattern.vertices[idx_boundary_sleeve_left_bottom]
    v_right_bottom = mesh_pattern.vertices[idx_boundary_sleeve_right_bottom]

    idx_cuff_left_l2h = np.argsort(v_cuff_left[:, 1])
    idx_cuff_right_l2h = np.argsort(v_cuff_right[:, 1])

    if v_left_up[:, 0].max() - v_cuff_left[idx_cuff_left_l2h[-1], 0] < 1e-6:
        idx_boundary_sleeve_left_up = np.append(idx_boundary_sleeve_left_up, idx_boundary_cuff_left[idx_cuff_left_l2h[-1]])
        
    if v_right_up[:, 0].min() - v_cuff_right[idx_cuff_right_l2h[-1], 0] > 1e-6:
        idx_boundary_sleeve_right_up = np.append(idx_boundary_sleeve_right_up, idx_boundary_cuff_right[idx_cuff_right_l2h[-1]])

    if len(v_left_bottom) and len(v_right_bottom):
        if v_left_bottom[:, 0].max() - v_cuff_left[idx_cuff_left_l2h[0], 0] < 1e-6:
            idx_boundary_sleeve_left_bottom = np.append(idx_boundary_sleeve_left_bottom, idx_boundary_cuff_left[idx_cuff_left_l2h[0]])
        if v_right_bottom[:, 0].max() - v_cuff_right[idx_cuff_right_l2h[0], 0] > 1e-6:
            idx_boundary_sleeve_right_bottom = np.append(idx_boundary_sleeve_right_bottom, idx_boundary_cuff_right[idx_cuff_right_l2h[0]])


        idx_closest_left_bottom = idx_boundary_sleeve_left_bottom[np.argmin(v_left_bottom[:, 0])]
        idx_closest_right_bottom = idx_boundary_sleeve_right_bottom[np.argmax(v_right_bottom[:, 0])]
    else:
        idx_boundary_sleeve_left_bottom = idx_boundary_sleeve_right_bottom = idx_closest_left_bottom = idx_closest_right_bottom = None

    return idx_boundary_sleeve_left_up, idx_boundary_sleeve_left_bottom, idx_boundary_sleeve_right_up, idx_boundary_sleeve_right_bottom, idx_closest_left_bottom, idx_closest_right_bottom


def get_skirt_idx(mesh_pattern, idx_boundary_v, label):
    idx_boundary_body_bottom = idx_boundary_v[label==0]
    idx_boundary_body_top = idx_boundary_v[label==2]
    
    idx_boundary_body_left = idx_boundary_v[label==1]
    idx_boundary_body_right = idx_boundary_v[label==3]
        
    v_bottom = mesh_pattern.vertices[idx_boundary_body_bottom]
    v_top = mesh_pattern.vertices[idx_boundary_body_top]
    v_left = mesh_pattern.vertices[idx_boundary_body_left]
    v_right = mesh_pattern.vertices[idx_boundary_body_right]

    idx_bottom_r2l = np.argsort(v_bottom[:, 0])
    if v_left[:, 1].min() - v_bottom[idx_bottom_r2l[-1], 1] > 1e-6:
        idx_boundary_body_left = np.append(idx_boundary_body_left, idx_boundary_body_bottom[idx_bottom_r2l[-1]])

    if v_right[:, 1].min() - v_bottom[idx_bottom_r2l[0], 1] > 1e-6:
        idx_boundary_body_right = np.append(idx_boundary_body_right, idx_boundary_body_bottom[idx_bottom_r2l[0]])

    idx_top_r2l = np.argsort(v_top[:, 0])
    if v_top[idx_top_r2l[-1], 1] - v_left[:, 1].max()  > 1e-6:
        idx_boundary_body_left = np.insert(idx_boundary_body_left, 0, idx_boundary_body_top[idx_top_r2l[-1]])

    if v_top[idx_top_r2l[0], 1] - v_right[:, 1].max() > 1e-6:
        idx_boundary_body_right = np.insert(idx_boundary_body_right, 0, idx_boundary_body_top[idx_top_r2l[0]])

    return idx_boundary_body_left, idx_boundary_body_right

def get_pants_idx(mesh_pattern, idx_boundary_v, label):
    idx_boundary_body_top = idx_boundary_v[label==5]
    idx_boundary_body_bottom_right = idx_boundary_v[label==0]
    idx_boundary_body_bottom_left = idx_boundary_v[label==3]
    
    idx_boundary_body_left_left = idx_boundary_v[label==4]
    idx_boundary_body_left_right = idx_boundary_v[label==2]
    idx_boundary_body_right_right = idx_boundary_v[label==6]
    idx_boundary_body_right_left = idx_boundary_v[label==1]
        
    v_bottom_r = mesh_pattern.vertices[idx_boundary_body_bottom_right]
    v_bottom_l = mesh_pattern.vertices[idx_boundary_body_bottom_left]
    v_top = mesh_pattern.vertices[idx_boundary_body_top]

    v_left_l = mesh_pattern.vertices[idx_boundary_body_left_left]
    v_left_r = mesh_pattern.vertices[idx_boundary_body_left_right]
    v_right_l = mesh_pattern.vertices[idx_boundary_body_right_left]
    v_right_r = mesh_pattern.vertices[idx_boundary_body_right_right]
    

    idx_bottom_l_r2l = np.argsort(v_bottom_l[:, 0])
    if v_left_l[:, 1].min() - v_bottom_l[idx_bottom_l_r2l[-1], 1] > 1e-6:
        idx_boundary_body_left_left = np.append(idx_boundary_body_left_left, idx_boundary_body_bottom_left[idx_bottom_l_r2l[-1]])
    if v_left_r[:, 1].min() - v_bottom_l[idx_bottom_l_r2l[0], 1] > 1e-6:
        idx_boundary_body_left_right = np.append(idx_boundary_body_left_right, idx_boundary_body_bottom_left[idx_bottom_l_r2l[0]])

    idx_bottom_r_r2l = np.argsort(v_bottom_r[:, 0])
    if v_right_r[:, 1].min() - v_bottom_r[idx_bottom_r_r2l[0], 1] > 1e-6:
        idx_boundary_body_right_right = np.append(idx_boundary_body_right_right, idx_boundary_body_bottom_right[idx_bottom_r_r2l[0]])
    if v_right_l[:, 1].min() - v_bottom_r[idx_bottom_r_r2l[-1], 1] > 1e-6:
        idx_boundary_body_right_left = np.append(idx_boundary_body_right_left, idx_boundary_body_bottom_right[idx_bottom_r_r2l[-1]])

    idx_top_r2l = np.argsort(v_top[:, 0])
    if v_top[idx_top_r2l[-1], 1] - v_left_l[:, 1].max()  > 1e-6:
        idx_boundary_body_left_left = np.insert(idx_boundary_body_left_left, 0, idx_boundary_body_top[idx_top_r2l[-1]])

    if v_top[idx_top_r2l[0], 1] - v_right_r[:, 1].max() > 1e-6:
        idx_boundary_body_right_right = np.insert(idx_boundary_body_right_right, 0, idx_boundary_body_top[idx_top_r2l[0]])

    idx_highest_left_right = idx_boundary_body_left_right[np.argmax(v_left_r[:, 1])]
    idx_highest_right_left = idx_boundary_body_right_left[np.argmax(v_right_l[:, 1])]

    return idx_boundary_body_left_left, idx_boundary_body_left_right, idx_boundary_body_right_right, idx_boundary_body_right_left, idx_highest_left_right, idx_highest_right_left


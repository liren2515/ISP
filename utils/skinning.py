import numpy as np
import torch
import torch.nn.functional as F

def infer_smpl(pose, beta, smpl_server):
    with torch.no_grad():
        output = smpl_server.forward_custom(betas=beta,
                                    #transl=transl,
                                    body_pose=pose[:, 3:],
                                    global_orient=pose[:, :3],
                                    return_verts=True,
                                    return_full_pose=True,
                                    v_template=smpl_server.v_template, rectify_root=False)
    w = output.weights
    tfs = output.T
    verts = output.vertices
    pose_offsets = output.pose_offsets
    shape_offsets = output.shape_offsets
    root_J = output.joints[:,[0]]

    return w, tfs, verts, pose_offsets, shape_offsets, root_J

def skinning_init(points, w_smpl, tfs, pose_offsets, shape_offsets, weights):
    _B = points.shape[0]
    #print(points.shape)
    
    Rot = torch.einsum('bnj,bjef->bnef', w_smpl, tfs)
    Rot_weighted = torch.einsum('bpn,bnij->bpij', weights, Rot)

    offsets = pose_offsets + shape_offsets
    offsets_weighted = torch.einsum('bpn,bni->bpi', weights, offsets)

    points = points + offsets_weighted

    points_h = F.pad(points, (0, 1), value=1.0)
    points_new = torch.einsum('bpij,bpj->bpi', Rot_weighted, points_h)
    points_new = points_new[:,:,:3]
    
    return points_new


def skinning_init_pants(points, w_smpl, tfs, pose_offsets, shape_offsets, weights, Rot_rest, pose_offsets_rest):
    _B = points.shape[0]
    #print(points.shape)

    Rot_rest_weighted_inv = torch.einsum('bpn,nij->bpij', weights, Rot_rest)
    Rot_rest_weighted_inv = torch.linalg.inv(Rot_rest_weighted_inv)
    offsets_rest_weighted = torch.einsum('bpn,ni->bpi', weights, pose_offsets_rest)

    Rot = torch.einsum('bnj,bjef->bnef', w_smpl, tfs)
    Rot_weighted = torch.einsum('bpn,bnij->bpij', weights, Rot)

    offsets = pose_offsets + shape_offsets
    offsets_weighted = torch.einsum('bpn,bni->bpi', weights, offsets)

    points_h = F.pad(points, (0, 1), value=1.0)
    points = torch.einsum('bpij,bpj->bpi', Rot_rest_weighted_inv, points_h)
    points = points[:,:,:3]
    points = points - offsets_rest_weighted

    points = points + offsets_weighted
    points_h = F.pad(points, (0, 1), value=1.0)
    points_new = torch.einsum('bpij,bpj->bpi', Rot_weighted, points_h)
    points_new = points_new[:,:,:3]
    
    return points_new
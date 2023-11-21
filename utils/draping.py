import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from utils.skinning import skinning_init, skinning_init_pants


def transform_pose(pose):
    pose = pose.cpu().numpy()
    root_rot = R.from_rotvec(pose[:, :3])
    rotate_original = root_rot.as_matrix()
    root_rot = root_rot.as_euler('zxy', degrees=True)
    root_rot[:, -1] = 0
    root_rot = R.from_euler('zxy', root_rot, degrees=True)
    rotate_zero = root_rot.as_matrix()
    rotate_zero_inv = np.linalg.inv(rotate_zero)
    root_rot = root_rot.as_rotvec()
    pose[:, :3] = root_rot
    pose = torch.FloatTensor(pose).cuda()

    return pose, rotate_original, rotate_zero_inv

def search_border_y(mask, x):
    mask_y = mask[x]
    y_l = 0
    y_r = 0
    flip = 0
    for i in range(1,len(mask_y)):
        if mask_y[i] != mask_y[i-1]:
            flip += 1
            if flip == 2:
                y_l = i-1
            elif flip == 3:
                y_r = i
    if flip != 4:
        raise ValueError('Somthing Wrong!!!!')
    return y_l, y_r

def generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv):
    # find pin points for top garments: collar near neck
    res = 200
    base_f = trimesh.proximity.ProximityQuery(mesh_pattern_f)

    _, dist_f, _ = base_f.on_surface(mesh_uv.vertices)
    thresh = 2./(res-1)*np.sqrt(2)
    unique_v_f = dist_f <= thresh

    mask_f = np.zeros((res, res)).reshape(-1)
    mask_f[unique_v_f] = 1
    mask_f = mask_f.reshape(res, res)

    mask_f_x = mask_f.sum(axis=-1)
    idx_f_x = np.where(mask_f_x>0) 
    x_s_f = np.min(idx_f_x) 

    y_l_f, y_r_f = search_border_y(mask_f, x_s_f)

    fix_mask = mask_f*0 + 1
    fix_mask[x_s_f:x_s_f+3,y_r_f-2:y_r_f+1] = 0
    fix_mask[x_s_f:x_s_f+3,y_l_f:y_l_f+3] = 0
    return fix_mask

def generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv):
    # find pin points for bottom garments: waist
    res = 200
    base_f = trimesh.proximity.ProximityQuery(mesh_pattern_f)

    _, dist_f, _ = base_f.on_surface(mesh_uv.vertices)
    thresh = 2./(res-1)*np.sqrt(2)
    unique_v_f = dist_f <= thresh

    mask_f = np.zeros((res, res)).reshape(-1)
    mask_f[unique_v_f] = 1
    mask_f = mask_f.reshape(res, res)

    mask_f_x = mask_f.sum(axis=-1)
    idx_f_x = np.where(mask_f_x>0) 
    x_s_f = np.min(idx_f_x) 

    fix_mask = mask_f*0 + 1
    fix_mask[x_s_f:x_s_f+3] = 0
    return fix_mask

def barycentric_faces(mesh_query, mesh_base, return_tensor=False):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    if return_tensor:
        v_barycentric = torch.FloatTensor(v_barycentric).cuda()
        closest_face_idx = torch.LongTensor(closest_face_idx).cuda()
    return v_barycentric, closest_face_idx


def prepare_barycentric_uv2atlas(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, mesh_uv, res=200, return_tensor=False):

    base_f = trimesh.proximity.ProximityQuery(mesh_pattern_f)
    base_b = trimesh.proximity.ProximityQuery(mesh_pattern_b)

    _, dist_f, _ = base_f.on_surface(mesh_uv.vertices)
    _, dist_b, _ = base_b.on_surface(mesh_uv.vertices)
    thresh = 2./(res-1)*np.sqrt(2)
    unique_v_f = dist_f <= thresh
    unique_v_b = dist_b <= thresh
    uv_v_f = mesh_uv.vertices[unique_v_f]
    uv_v_b = mesh_uv.vertices[unique_v_b]

    closest_pt_f, _, closest_face_idx_f = base_f.on_surface(uv_v_f)
    closest_pt_b, _, closest_face_idx_b = base_b.on_surface(uv_v_b)
    triangles_f = mesh_pattern_f.triangles[closest_face_idx_f]
    triangles_b = mesh_pattern_b.triangles[closest_face_idx_b]
    v_barycentric_pattern_f = trimesh.triangles.points_to_barycentric(triangles_f, closest_pt_f)
    v_barycentric_pattern_b = trimesh.triangles.points_to_barycentric(triangles_b, closest_pt_b)
    
    closest_atlas_f = trimesh.triangles.barycentric_to_points(mesh_atlas_f.triangles[closest_face_idx_f], v_barycentric_pattern_f)
    closest_atlas_b = trimesh.triangles.barycentric_to_points(mesh_atlas_b.triangles[closest_face_idx_b], v_barycentric_pattern_b)
    
    indicator_unique_v_f = np.zeros((len(mesh_uv.vertices)))
    indicator_unique_v_b = np.zeros((len(mesh_uv.vertices)))
    indicator_unique_v_f[unique_v_f] = 1
    indicator_unique_v_b[unique_v_b] = 1
    
    input_pattern_f = np.zeros((200,200,3)).reshape(-1,3)-1
    input_pattern_b = np.zeros((200,200,3)).reshape(-1,3)-1
    input_pattern_f[unique_v_f] = closest_atlas_f
    input_pattern_b[unique_v_b] = closest_atlas_b
    input_pattern_f = input_pattern_f.reshape(200,200,3)
    input_pattern_b = input_pattern_b.reshape(200,200,3)

    if return_tensor:
        v_barycentric_pattern_f = torch.FloatTensor(v_barycentric_pattern_f).cuda()
        v_barycentric_pattern_b = torch.FloatTensor(v_barycentric_pattern_b).cuda()
        closest_face_idx_f = torch.LongTensor(closest_face_idx_f).cuda()
        closest_face_idx_b = torch.LongTensor(closest_face_idx_b).cuda()
        input_pattern_f = torch.FloatTensor(input_pattern_f).cuda().permute(2,0,1)
        input_pattern_b = torch.FloatTensor(input_pattern_b).cuda().permute(2,0,1)
        indicator_unique_v_f = torch.BoolTensor(indicator_unique_v_f).cuda()
        indicator_unique_v_b = torch.BoolTensor(indicator_unique_v_b).cuda()

    return v_barycentric_pattern_f, v_barycentric_pattern_b, closest_face_idx_f, closest_face_idx_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b


def uv_to_3D(pattern_deform, barycentric_uv_batch, closest_face_idx_uv_batch, uv_faces_cuda):
    _B = len(closest_face_idx_uv_batch)
    uv_faces_id = uv_faces_cuda[closest_face_idx_uv_batch.reshape(-1)].reshape(_B, -1, 3)
    uv_faces_id = uv_faces_id.reshape(_B, -1)
    uv_faces_id = uv_faces_id.unsqueeze(-1).repeat(1,1,3).detach().cuda()

    pattern_deform_triangles = torch.gather(pattern_deform, 1, uv_faces_id)
    pattern_deform_triangles = pattern_deform_triangles.reshape(_B, -1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv_batch[:, :, :, None]).sum(dim=-2)
    return pattern_deform_bary

    
def draping(packed_input, pose, beta, model_diffusion, model_draping, uv_faces, packed_input_smpl, is_pants=False, Rot_rest=None, pose_offsets_rest=None):
    # vertices_garment_T - (#P, 3) 
    vertices_garment_T_f, vertices_garment_T_b, faces_garment_f, faces_garment_b, barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code = packed_input
    with torch.no_grad():
        num_v_f = len(vertices_garment_T_f)
        num_v_b = len(vertices_garment_T_b)
        barycentric_uv_batch_f = barycentric_uv_f.unsqueeze(0)
        barycentric_uv_batch_b = barycentric_uv_b.unsqueeze(0)
        closest_face_idx_uv_batch_f = closest_face_idx_uv_f.unsqueeze(0)
        closest_face_idx_uv_batch_b = closest_face_idx_uv_b.unsqueeze(0)

        fix_mask = fix_mask.unsqueeze(0)
    
        pose = pose.unsqueeze(0)
        beta = beta.unsqueeze(0)
        latent_code = latent_code.unsqueeze(0)

        points = torch.cat((vertices_garment_T_f, vertices_garment_T_b), dim=0)
        
        w_smpl, tfs, pose_offsets, shape_offsets = packed_input_smpl
        weight_points = model_diffusion(points*10)
        weight_points = F.softmax(weight_points, dim=-1)
        weight_points = weight_points.reshape(1, -1, 6890)
        if is_pants:
            garment_skinning_init = skinning_init_pants(points.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_points, Rot_rest, pose_offsets_rest)
        else:
            garment_skinning_init = skinning_init(points.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_points)
        
        smpl_param = torch.cat((pose, beta, latent_code), dim=-1)
        pattern_deform_f, pattern_deform_b = model_draping(smpl_param)
        pattern_deform_f = pattern_deform_f * fix_mask[:, None]
        pattern_deform_b = pattern_deform_b * fix_mask[:, None]


        pattern_deform_f = pattern_deform_f.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_b = pattern_deform_b.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_bary_f = uv_to_3D(pattern_deform_f, barycentric_uv_batch_f, closest_face_idx_uv_batch_f, uv_faces)
        pattern_deform_bary_b = uv_to_3D(pattern_deform_b, barycentric_uv_batch_b, closest_face_idx_uv_batch_b, uv_faces)

        garment_skinning_init[:, :num_v_f] += pattern_deform_bary_f#.squeeze(0)
        garment_skinning_init[:, num_v_f:] += pattern_deform_bary_b#.squeeze(0)
        garment_skinning = garment_skinning_init

        #if is_underlying:
        #    garment_faces = torch.LongTensor(np.concatenate((faces_garment_f, num_v_f+faces_garment_b), axis=0)).unsqueeze(0)
        #    deformed_cloth.update_single(garment_skinning, garment_faces)


    garment_skinning_f = garment_skinning[:, :num_v_f]
    garment_skinning_b = garment_skinning[:, num_v_f:]
    garment_faces = torch.LongTensor(np.concatenate((faces_garment_f, num_v_f+faces_garment_b), axis=0)).unsqueeze(0)
    
    return garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces


def draping_grad(packed_input, pose, beta, model_diffusion, model_draping, uv_faces, packed_input_smpl, is_pants=False, Rot_rest=None, pose_offsets_rest=None):
    # vertices_garment_T - (#P, 3) 
    vertices_garment_T_f, vertices_garment_T_b, faces_garment_f, faces_garment_b, barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code = packed_input
    with torch.no_grad():
        num_v_f = len(vertices_garment_T_f)
        num_v_b = len(vertices_garment_T_b)
        barycentric_uv_batch_f = barycentric_uv_f.unsqueeze(0)
        barycentric_uv_batch_b = barycentric_uv_b.unsqueeze(0)
        closest_face_idx_uv_batch_f = closest_face_idx_uv_f.unsqueeze(0)
        closest_face_idx_uv_batch_b = closest_face_idx_uv_b.unsqueeze(0)

        fix_mask = fix_mask.unsqueeze(0)
    
        pose = pose.unsqueeze(0)
        beta = beta.unsqueeze(0)
        latent_code = latent_code.unsqueeze(0)

        points = torch.cat((vertices_garment_T_f, vertices_garment_T_b), dim=0)
        
        w_smpl, tfs, pose_offsets, shape_offsets = packed_input_smpl
        weight_points = model_diffusion(points*10)
        weight_points = F.softmax(weight_points, dim=-1)
        weight_points = weight_points.reshape(1, -1, 6890)

    if is_pants:
        garment_skinning_init = skinning_init_pants(points.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_points, Rot_rest, pose_offsets_rest)
    else:
        garment_skinning_init = skinning_init(points.unsqueeze(0), w_smpl, tfs, pose_offsets, shape_offsets, weight_points)
        
    with torch.no_grad():
        smpl_param = torch.cat((pose, beta, latent_code), dim=-1)
        pattern_deform_f, pattern_deform_b = model_draping(smpl_param)
        pattern_deform_f = pattern_deform_f * fix_mask[:, None]
        pattern_deform_b = pattern_deform_b * fix_mask[:, None]


        pattern_deform_f = pattern_deform_f.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_b = pattern_deform_b.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_bary_f = uv_to_3D(pattern_deform_f, barycentric_uv_batch_f, closest_face_idx_uv_batch_f, uv_faces)
        pattern_deform_bary_b = uv_to_3D(pattern_deform_b, barycentric_uv_batch_b, closest_face_idx_uv_batch_b, uv_faces)

    garment_skinning_init[:, :num_v_f] += pattern_deform_bary_f
    garment_skinning_init[:, num_v_f:] += pattern_deform_bary_b
    garment_skinning = garment_skinning_init

    garment_skinning_f = garment_skinning[:, :num_v_f]
    garment_skinning_b = garment_skinning[:, num_v_f:]
    garment_faces = torch.LongTensor(np.concatenate((faces_garment_f, num_v_f+faces_garment_b), axis=0)).unsqueeze(0)
    
    return garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces
import torch
import trimesh
import numpy as np
import os, sys
import cv2
import random

from smpl_pytorch.body_models import SMPL
from networks import drape, SDF, unet
from utils import mesh_reader
from utils.snug_class import Deformed_Cloth, Body
from utils.ISP import reconstruct_batch
from utils.draping import generate_fix_mask, generate_fix_mask_bottom, barycentric_faces, prepare_barycentric_uv2atlas, transform_pose, draping
from utils.layering import draping_layer
from utils.skinning import infer_smpl
from utils.fitting import resolve_collision


def load_smpl_server(gender='f'):
    smpl_server = SMPL(model_path='./smpl_pytorch',
                                gender=gender,
                                use_hands=False,
                                use_feet_keypoints=False,
                                dtype=torch.float32).cuda()
    
    smpl_body = Body(smpl_server.faces)

    pose = torch.zeros(1, 72).cuda()
    beta = torch.zeros(1, 10).cuda()
    pose = pose.reshape(24,3)
    pose[1, 2] = .35
    pose[2, 2] = -.35
    pose = pose.reshape(-1).unsqueeze(0)
    w, tfs, _, pose_offsets, _, _ = infer_smpl(pose, beta, smpl_server)
    Rot_rest = torch.einsum('nk,kij->nij', w.squeeze(), tfs.squeeze()) 
    pose_offsets_rest = pose_offsets.squeeze()

    return smpl_server, smpl_server.faces, smpl_body, Rot_rest, pose_offsets_rest

def load_template_model(res=200, is_pants=True, data_path='./extra-data', ckpt_path='./checkpoints'):
    statistic = np.load(os.path.join(data_path, 'shirt.npz'))
    y_center = statistic['y_center']
    diag_max = statistic['diag_max']

    rep_size = 32
    model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+11, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+10, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_rep = SDF.learnt_representations(rep_size=rep_size, samples=400).cuda()
    model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()

    model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_sdf_f.pth')))
    model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_sdf_b.pth')))
    model_rep.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_rep.pth')))

    model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_atlas_f.pth')))
    model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_atlas_b.pth')))

    x_res = y_res = res
    uv_vertices, uv_faces = mesh_reader.create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
    edges = torch.LongTensor(mesh_uv.edges).cuda()
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()

    latent_codes = model_rep.weights.detach()

    model_atlas_f_bottom = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_atlas_b_bottom = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    if is_pants:
        model_sdf_f_bottom = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+7, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_sdf_b_bottom = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+7, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_rep_bottom = SDF.learnt_representations(rep_size=rep_size, samples=200).cuda()

        model_sdf_f_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_sdf_f.pth')))
        model_sdf_b_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_sdf_b.pth')))
        model_rep_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_rep.pth')))

        model_atlas_f_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_atlas_f.pth')))
        model_atlas_b_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_atlas_b.pth')))
    else:
        model_sdf_f_bottom = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+4, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_sdf_b_bottom = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+4, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_rep_bottom = SDF.learnt_representations(rep_size=rep_size, samples=300).cuda()

        model_sdf_f_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_sdf_f.pth')))
        model_sdf_b_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_sdf_b.pth')))
        model_rep_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_rep.pth')))

        model_atlas_f_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_atlas_f.pth')))
        model_atlas_b_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_atlas_b.pth')))

    latent_codes_bottom = model_rep_bottom.weights.detach()

    if is_pants:
        statistic_bottom = np.load(os.path.join(data_path, 'pants.npz'))
    else:
        statistic_bottom = np.load(os.path.join(data_path, 'skirt.npz'))
    y_center_bottom = statistic_bottom['y_center']
    diag_max_bottom = statistic_bottom['diag_max']

    return model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, latent_codes, model_sdf_f_bottom, model_sdf_b_bottom, model_rep_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_codes_bottom, mesh_uv, uv_vertices, uv_faces, edges, y_center, diag_max, y_center_bottom, diag_max_bottom

def load_draping_model(is_pants=True, ckpt_path='./checkpoints'):
    model_draping = drape.Pred_decoder_uv_linear(d_in=82+32, d_hidden=512, depth=8, skip_layer=[4], tanh=False).cuda()
    model_draping.load_state_dict(torch.load(os.path.join(ckpt_path, 'drape_shirt.pth')))
    
    model_diffusion = drape.skip_connection(d_in=3, width=512, depth=8, d_out=6890, skip_layer=[]).cuda()
    model_diffusion.load_state_dict(torch.load(os.path.join(ckpt_path, 'smpl_diffusion.pth')))
    
    model_layer = unet.UNet_isolateNode(in_channels=18, out_channels=6, init_features=100).cuda()
    model_layer.load_state_dict(torch.load(os.path.join(ckpt_path, 'layering.pth')))

    deformed_cloth = Deformed_Cloth()

    model_draping_bottom = drape.Pred_decoder_uv_linear(d_in=82+32, d_hidden=512, depth=8, skip_layer=[4], tanh=False).cuda()
    if is_pants:
        model_draping_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'drape_pants.pth')))
    else:
        model_draping_bottom.load_state_dict(torch.load(os.path.join(ckpt_path, 'drape_skirt.pth')))
    
    model_diffusion_TA = drape.skip_connection(d_in=3, width=512, depth=8, d_out=6890, skip_layer=[]).cuda()
    model_diffusion_TA.load_state_dict(torch.load(os.path.join(ckpt_path, 'smpl_diffusion_TA.pth')))

    return model_draping, model_diffusion, model_draping_bottom, model_diffusion_TA, model_layer, deformed_cloth



is_pants = True
is_sewing = True

smpl_server, faces_body, smpl_body, Rot_rest, pose_offsets_rest = load_smpl_server()

poses = torch.load(os.path.join('/cvlabdata2/home/ren/Public-Release/ISP/extra-data/', 'pose-sample.pt'))
pose = poses[[0]].cuda()
pose, rotate_original, rotate_zero_inv = transform_pose(pose)
pose = pose.squeeze()
beta = torch.zeros(10).cuda()

uv_vertices_200, uv_faces_200 = mesh_reader.create_uv_mesh(200, 200, debug=False)
mesh_uv_200 = trimesh.Trimesh(uv_vertices_200, uv_faces_200, process=False, validate=False)

x_res = y_res = 180
model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, latent_codes, model_sdf_f_bottom, model_sdf_b_bottom, model_rep_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_codes_bottom, mesh_uv, uv_vertices, uv_faces, edges, y_center, diag_max, y_center_bottom, diag_max_bottom = load_template_model(res=x_res, is_pants=is_pants)
model_draping, model_diffusion, model_draping_bottom, model_diffusion_TA, model_layer, deformed_cloth = load_draping_model(is_pants=is_pants)

save_dir = os.path.join('tmp/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


############# infer SMPL #############
with torch.no_grad():
    w_smpl, tfs, verts_body, pose_offsets, shape_offsets, root_J = infer_smpl(pose.unsqueeze(0), beta.unsqueeze(0), smpl_server)
    root_J = root_J.squeeze(0).cpu().numpy()
    packed_input_smpl = [w_smpl, tfs, pose_offsets, shape_offsets]
    smpl_body.update_body(verts_body)
    mesh_body = trimesh.Trimesh(verts_body[0].detach().cpu().numpy(), faces_body)
    mesh_body.vertices = np.einsum('ij,nj->ni', rotate_original[0], np.einsum('ij,nj->ni', rotate_zero_inv[0], mesh_body.vertices - root_J)) + root_J
    
    mesh_body.export(os.path.join(save_dir, 'body.obj'))


############# infer ISP #############
uv_faces_cuda = torch.LongTensor(uv_faces_200).cuda()
idx_list = [8, 1, 2, 3]
packed_input = []
packed_input_atlas = []
faces_sewing = []
for i in range(len(idx_list)):
    print('Reconstruct ', i)
    idx_G = idx_list[i]
    if i == 0:
        which = 'pants' if is_pants else 'skirt'
        latent_code = latent_codes_bottom[idx_G]

        mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = reconstruct_batch(model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_code, uv_vertices, uv_faces, edges, resolution=x_res, which=which)

        fix_mask = generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

    else:
        latent_code = latent_codes[idx_G]

        mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code, uv_vertices, uv_faces, edges, resolution=x_res, which='tee')

        fix_mask = generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

    faces_sewing.append(mesh_sewing.faces)
    fix_mask = torch.FloatTensor(fix_mask).cuda()

    barycentric_uv_f, closest_face_idx_uv_f = barycentric_faces(mesh_pattern_f, mesh_uv_200, return_tensor=True)
    barycentric_uv_b, closest_face_idx_uv_b = barycentric_faces(mesh_pattern_b, mesh_uv_200, return_tensor=True)

    if i != 0:
        barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b = prepare_barycentric_uv2atlas(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, mesh_uv_200, res=200, return_tensor=True)

        packed_input_atlas_i = [barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b]
    else:
        packed_input_atlas_i = []

    # for SMPL body alignment
    if i == 0:
        mesh_sewing.vertices = mesh_sewing.vertices*diag_max_bottom/2 
        mesh_sewing.vertices[:, 1] += y_center_bottom
        mesh_atlas_f.vertices = mesh_atlas_f.vertices*diag_max_bottom/2 
        mesh_atlas_b.vertices = mesh_atlas_b.vertices*diag_max_bottom/2 
        mesh_atlas_f.vertices[:, 1] += y_center_bottom
        mesh_atlas_b.vertices[:, 1] += y_center_bottom
    else:
        mesh_sewing.vertices = mesh_sewing.vertices*diag_max/2 
        mesh_sewing.vertices[:, 1] += y_center
        mesh_atlas_f.vertices = mesh_atlas_f.vertices*diag_max/2 
        mesh_atlas_b.vertices = mesh_atlas_b.vertices*diag_max/2 
        mesh_atlas_f.vertices[:, 1] += y_center
        mesh_atlas_b.vertices[:, 1] += y_center

    vertices_garment_T_f = torch.FloatTensor(mesh_atlas_f.vertices).cuda()
    vertices_garment_T_b = torch.FloatTensor(mesh_atlas_b.vertices).cuda()
    faces_garment_f = mesh_atlas_f.faces
    faces_garment_b = mesh_atlas_b.faces
    
    packed_input_i = [vertices_garment_T_f, vertices_garment_T_b, faces_garment_f, faces_garment_b, barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code]

    packed_input.append(packed_input_i)
    packed_input_atlas.append(packed_input_atlas_i)


############# draping #############
packed_skinning = []
for i in range(len(idx_list)):
    if i == 0:
        model_diffusion_which = model_diffusion_TA if is_pants else model_diffusion
        garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = draping(packed_input[i], pose, beta, model_diffusion_which, model_draping_bottom, uv_faces_cuda, packed_input_smpl, is_pants=is_pants, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)
    else:
        garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = draping(packed_input[i], pose, beta, model_diffusion, model_draping, uv_faces_cuda, packed_input_smpl)
    packed_skinning_i = [garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces]

    packed_skinning.append(packed_skinning_i)


############# layering #############
for i in range(len(idx_list)):
    is_bottom = i==0
    for j in range(i+1, len(idx_list)):
        print('Layering ', i, j)
        
        
        garment_layer_f_j, garment_layer_b_j = draping_layer(packed_input[i], packed_input[j], packed_input_atlas[j], packed_skinning[i], packed_skinning[j], pose, beta, model_diffusion, model_draping, model_layer, uv_faces_cuda, deformed_cloth, body=smpl_body, is_bottom=is_bottom)

        garment_skinning_f_j = garment_layer_f_j
        garment_skinning_b_j = garment_layer_b_j
        garment_skinning_j = torch.cat((garment_layer_f_j, garment_layer_b_j), dim=1)
        packed_skinning[j] = [garment_skinning_f_j, garment_skinning_b_j, garment_skinning_j, packed_skinning[j][-1]]

############# saving mesh #############
color_map = [[100, 100, 100],
            [255, 51, 51],
            [255, 153, 51],
            [255, 255, 51],
            [153, 255, 51],
            [51, 255, 51],
            [51, 255, 153],
            [51, 255, 255],
            [51, 153, 255],
            [51, 51, 255],
            [153, 51, 255]]
color_map = np.array(color_map).astype(int)

for i in range(len(idx_list)):
    print('Saving ', i)
    garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = packed_skinning[i]
    if is_sewing:
        mesh_layer = trimesh.Trimesh(garment_skinning.squeeze().cpu().numpy(), faces_sewing[i], process=False, validate=False)
    else:
        mesh_layer = trimesh.Trimesh(garment_skinning.squeeze().cpu().numpy(), garment_faces.squeeze().cpu().numpy(), process=False, validate=False)
    mesh_layer.vertices = np.einsum('ij,nj->ni', rotate_original[0], np.einsum('ij,nj->ni', rotate_zero_inv[0], mesh_layer.vertices - root_J)) + root_J

    #if i<2:
    #    mesh_layer = resolve_collision(mesh_layer, mesh_body, scale=2)

    colors_v = np.ones((len(mesh_layer.vertices), 3))*color_map[i][np.newaxis,:]
    mesh_layer.visual.vertex_colors = colors_v
    #if i!= 0 :
    trimesh.smoothing.filter_taubin(mesh_layer, lamb=0.5)

    mesh_gar_layer_name = 'layer_bottom_%03d.obj'%i
    mesh_layer.export(os.path.join(save_dir, mesh_gar_layer_name), include_color=True)

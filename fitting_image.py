import torch
import torch.nn.functional as F
import trimesh
import numpy as np
import os, sys
import cv2
import pickle as pkl

from smpl_pytorch.body_models import SMPL
from networks import drape, SDF, unet
from utils.snug_class import Deformed_Cloth, Body
from utils.skinning import infer_smpl
from utils import mesh_reader

from utils.fitting import parse_segmentation, process_segmentation, match_pose, get_render, fit_neutral2female, retrieve_cloth, fitting, prepare_draping_top, retrieve_cloth_layering, fitting_layering, infer


def load_smpl_server():
    smpl_server = SMPL(model_path='./smpl_pytorch',
                                gender='f',
                                use_hands=False,
                                use_feet_keypoints=False,
                                dtype=torch.float32).cuda()

    smpl_server_n = SMPL(model_path='./smpl_pytorch',
                                gender='neutral',
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

    return smpl_server, smpl_server_n, smpl_body, Rot_rest, pose_offsets_rest

def load_template_model(res=200, is_pants=True, data_path='./extra-data', ckpt_path='./checkpoints'):
    statistic = np.load(os.path.join(data_path, 'shirt.npz'))
    y_center = statistic['y_center'].item()
    diag_max = statistic['diag_max'].item()

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
    y_center_bottom = statistic_bottom['y_center'].item()
    diag_max_bottom = statistic_bottom['diag_max'].item()

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


root = '/cvlabdata2/home/ren/Public-Release/ISP-code/extra-data/fitting-sample/'
seg = cv2.imread(os.path.join(root, 'mask.png'), -1)
smpl_pred_path = os.path.join(root, 'mocap.pkl')

save_folder = os.path.join('tmp/fitting')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

smpl_pred = pkl.load(open(smpl_pred_path, 'rb'))
smpl_pred = smpl_pred['pred_output_list'][0]
topleft = smpl_pred['bbox_top_left']
scale_ratio = smpl_pred['bbox_scale_ratio']
pred_joints_img = smpl_pred['pred_joints_img'] #[49, 3]

images_gt = process_segmentation(seg, topleft, scale_ratio, save=True, save_folder=save_folder).cuda()

camScale = torch.FloatTensor(smpl_pred['pred_camera'])[0].cuda()
camTrans = torch.FloatTensor(smpl_pred['pred_camera'][1:]).cuda()
pose = torch.FloatTensor(smpl_pred['pred_body_pose']).cuda() #(1, 72)
beta = torch.FloatTensor(smpl_pred['pred_betas']).cuda() #(1, 72)

renderer_textured = get_render(camScale, camTrans)

smpl_server, smpl_server_n, body, Rot_rest, pose_offsets_rest = load_smpl_server()
faces_body = smpl_server.faces


############# fitting body #############
# fit from neutral body to female body
pose, shape, verts_pred, verts_gt = fit_neutral2female(pose, beta, smpl_server, smpl_server_n, lr=0.01, iters=1000)
body_parameter = torch.cat([pose, shape], dim=-1)
torch.save(body_parameter.cpu().detach(), save_folder + '/body-paramter-f.pt')

body_f = trimesh.Trimesh(verts_pred, faces_body)
body_n = trimesh.Trimesh(verts_gt, faces_body)
body_f.export(save_folder + '/body_f.obj')
body_n.export(save_folder + '/body_n.obj')

uv_vertices_200, uv_faces_200 = mesh_reader.create_uv_mesh(200, 200, debug=False)
mesh_uv_200 = trimesh.Trimesh(uv_vertices_200, uv_faces_200, process=False, validate=False)

x_res = y_res = 180
model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, latent_codes, model_sdf_f_bottom, model_sdf_b_bottom, model_rep_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_codes_bottom, mesh_uv, uv_vertices, uv_faces, edges, y_center, diag_max, y_center_bottom, diag_max_bottom = load_template_model(res=x_res, is_pants=True)
model_draping, model_diffusion, model_draping_bottom, model_diffusion_TA, model_layer, deformed_cloth = load_draping_model(is_pants=True)


#body_parameter = torch.load(save_folder + '/body-paramter-f.pt')
#pose = body_parameter[0, :72].cuda()
#beta = body_parameter[0, 72:].cuda()
pose = pose[0]
beta = shape[0]

pose, rotate_mat = match_pose(pose)

w_smpl, tfs, verts_body, pose_offsets, shape_offsets, root_J = infer_smpl(pose.unsqueeze(0), beta.unsqueeze(0), smpl_server)
packed_input_smpl = [w_smpl, tfs, pose_offsets, shape_offsets]
packed_body = [verts_body.squeeze(), torch.LongTensor(smpl_server.faces.astype(int)), root_J[0]]



############# fitting jacket #############
best_i = retrieve_cloth(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, packed_body, rotate_mat, save_folder, cloth_type=1)
print(best_i)

fitting(best_i, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, packed_body, rotate_mat, body, save_folder, cloth_type=1, thresh=-5e-2)

############# fitting tee #############
latent_code_jacket = torch.load(save_folder + '/best-z-fit-1.pt').cuda()
latent_code_top = [latent_code_jacket]
packed_top = prepare_draping_top(latent_code_top, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, uv_vertices, uv_faces, edges, mesh_uv_200, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, resolution=x_res)

best_i = retrieve_cloth_layering(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=0, is_pants=False)
print(best_i)

fitting_layering(best_i, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=0, is_pants=False)


############# fitting trousers #############
latent_code_tee = torch.load(save_folder + '/best-z-fit-0.pt').cuda()
latent_code_top = [latent_code_jacket, latent_code_tee]
packed_top = prepare_draping_top(latent_code_top, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, uv_vertices, uv_faces, edges, mesh_uv_200, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, resolution=x_res)

best_i = retrieve_cloth_layering(model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_codes_bottom, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max_bottom, y_center_bottom, pose, beta, model_diffusion_TA, model_draping_bottom, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=2, is_pants=True, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)
print(best_i)

fitting_layering(best_i, model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_codes_bottom, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max_bottom, y_center_bottom, pose, beta, model_diffusion_TA, model_draping_bottom, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=2, is_pants=True, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)


############# infer #############
latent_code_jacket = torch.load(save_folder + '/best-z-fit-1.pt').cuda()
latent_code_tee = torch.load(save_folder + '/best-z-fit-0.pt').cuda()
latent_code_trousers = torch.load(save_folder + '/best-z-fit-2.pt').cuda()
latent_codes = [latent_code_trousers, latent_code_tee, latent_code_jacket]
bottom = [model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, model_draping_bottom, diag_max_bottom, y_center_bottom]
top = [model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, model_draping, diag_max, y_center]

infer(latent_codes, top, bottom, model_diffusion, model_diffusion_TA, model_layer, uv_vertices, uv_faces, edges, mesh_uv_200, packed_input_smpl, packed_body, rotate_mat, pose, beta, body, deformed_cloth, save_folder, resolution=x_res, is_pants=True, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)

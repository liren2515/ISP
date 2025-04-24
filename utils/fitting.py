import os
import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import cv2
from pytorch3d.structures import Meshes
from utils.render import SimpleShader
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesVertex
)

from utils import mesh_reader, draping, ISP, layering

def parse_segmentation(segmentaion):
    seg = segmentaion.copy()[:,:,0]*0
    silh = seg.copy()
    idx_tee = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 128)
    idx_jacket = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 128) * (segmentaion[:,:,2] == 128)
    idx_pants = (segmentaion[:,:,0] == 0) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 192)
    idx_dress = (segmentaion[:,:,0] == 128) * (segmentaion[:,:,1] == 0) * (segmentaion[:,:,2] == 64)
    idx_silh = (segmentaion[:,:,0] > 0) + (segmentaion[:,:,1] > 0) + (segmentaion[:,:,2] > 0)
    seg[idx_tee] = 1
    seg[idx_jacket] = 2
    seg[idx_pants] = 3
    seg[idx_dress] = 4
    silh[idx_silh] = 1
    return seg, silh

def process_segmentation(seg_raw, topleft, scale_ratio, save=True, save_folder=None):
    # 0: background
    # 128/0/128:   tee
    # 128/128/128: jacket
    # 0/0/192:     pants
    # 128/0/64:    dress  

    padding = 400
    seg_large = np.zeros((seg_raw.shape[0]+padding, seg_raw.shape[1]+padding, 3)).astype(np.uint8)
    seg_large[padding//2:padding//2+seg_raw.shape[0], padding//2:padding//2+seg_raw.shape[1]] = seg_raw

    offset = 0
    min_y = int(topleft[1]) + padding//2
    max_y = int(topleft[1] + 224/scale_ratio) + padding//2 
    min_x = int(topleft[0]) + padding//2
    max_x = int(topleft[0] + 224/scale_ratio) + padding//2
    seg_cropped = seg_large[min_y:max_y, min_x:max_x]

    seg_gt, silh_gt = parse_segmentation(seg_cropped)

    dim = [224, 224]
    seg_gt = cv2.resize(seg_gt, dim)
    silh_gt = cv2.resize(silh_gt, dim)

    mask_tee = np.zeros((224, 224))
    mask_jacket = np.zeros((224, 224))
    mask_pants = np.zeros((224, 224))
    mask_dress = np.zeros((224, 224))
    mask_silh = np.zeros((224, 224))

    mask_tee[seg_gt==1] = 1
    mask_jacket[seg_gt==2] = 1
    mask_pants[seg_gt==3] = 1
    mask_dress[seg_gt==4] = 1
    mask_silh[silh_gt==1] = 1
    images_gt = torch.FloatTensor(np.stack([mask_tee, mask_jacket, mask_pants, mask_dress, mask_silh], axis=-1).astype(int))
    return images_gt

def match_pose(pose, small_hand=True, flip=True):
    from scipy.spatial.transform import Rotation as R
    pose = pose.unsqueeze(0)
    pose = pose.cpu().numpy()
    if flip:
        swap_rotation = R.from_euler('z', [180], degrees=True)
        root_rot = R.from_rotvec(pose[:, :3])
        pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    if small_hand:
        pose[:, 66:] *= 0.1

    root_rot = R.from_rotvec(pose[:, :3])
    rotate_original = root_rot.as_matrix()[0]
    root_rot = root_rot.as_euler('zxy', degrees=True)
    root_rot[:, -1] = 0
    root_rot = R.from_euler('zxy', root_rot, degrees=True)
    rotate_zero = root_rot.as_matrix()
    rotate_zero_inv = np.linalg.inv(rotate_zero)[0]
    root_rot = root_rot.as_rotvec()
    pose[:, :3] = root_rot
    pose = torch.FloatTensor(pose).cuda().squeeze()
    rotate_mat = torch.FloatTensor(rotate_original@rotate_zero_inv).cuda()
    return pose, rotate_mat


def get_render(camScale, camTrans):
    device = torch.device("cuda:0")

    render_res = 224
    focal_length = 5000
    tx = (camTrans[0]/camScale).item()
    ty = (camTrans[1]/camScale).item()
    T = torch.FloatTensor([-tx, -ty, 2*focal_length/(render_res*camScale)]).unsqueeze(0)
    cameras = PerspectiveCameras(device=device, T=T, focal_length=focal_length, in_ndc=False, image_size=[[render_res,render_res]], principal_point=((render_res/2, render_res/2),))

    sigma = 1e-5
    raster_settings_soft = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=20, 
        max_faces_per_bin=500000,
        perspective_correct=False
    )

    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings_soft
        ),
        shader=SimpleShader(device=device, hard_mode=False)
    )

    return renderer_textured



def resolve_collision(mesh_pants, mesh_body, scale=1.15):
    
    garment_skinning = torch.FloatTensor(mesh_pants.vertices).unsqueeze(0).cuda()
    vb = torch.FloatTensor(mesh_body.vertices).unsqueeze(0).cuda()
    nb = torch.FloatTensor(mesh_body.vertex_normals).unsqueeze(0).cuda()
    
    vec = garment_skinning[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(garment_skinning - vb)).sum(dim=-1) 
    garment_skinning[distance<=0] -= nb[distance<=0]*distance[distance<=0][:, None]*scale
    
    cloth_mesh = trimesh.Trimesh(garment_skinning.squeeze().cpu().numpy(), mesh_pants.faces)
    cloth_mesh.visual.face_colors = mesh_pants.visual.face_colors

    return cloth_mesh

def collision_penalty(va, vb, nb, eps=2e-3, kcollision=2500):
    batch_size = va.shape[0]
    '''
    closest_vertices = NearestNeighbour(dtype=va.dtype)(va, vb)
    vb = tf.gather(vb, closest_vertices, batch_dims=1)
    nb = tf.gather(nb, closest_vertices, batch_dims=1)

    distance = tf.reduce_sum(nb*(va - vb), axis=-1) 
    interpenetration = tf.maximum(eps - distance, 0)
    '''
    vec = va[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(va - vb)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)

    return (interpenetration**3).sum() / batch_size * kcollision

def fit_neutral2female(pose, shape, smpl_server, smpl_server_n, lr=0.01, iters=1000):

    transl = torch.zeros(1,3).cuda()
    transl.requires_grad = False
    with torch.no_grad():
        verts_gt, _, _, _, _ = smpl_server_n.forward_verts(betas=shape,
                                        transl=transl,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server_n.v_template)
                                        

    pose.requires_grad = True
    shape.requires_grad = True
    verts_gt.requires_grad = False

    optimizer = torch.optim.Adam([shape, pose], lr=lr)
    for i in range(iters):
        verts_pred, _, _, _, _ = smpl_server.forward_verts(betas=shape,
                                        transl=transl,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server.v_template)

        loss = (verts_gt-verts_pred.squeeze()).norm(2, dim=-1).mean()
        
        if i%100 == 0:
            print('iter: %3d: %0.5f'%(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        verts_pred, _, _, _, _ = smpl_server.forward_verts(betas=shape,
                                        transl=transl,
                                        body_pose=pose[:, 3:],
                                        global_orient=pose[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=smpl_server.v_template, rectify_root=False)
                                        
        loss = (verts_gt-verts_pred.squeeze()).norm(2, dim=-1).mean()
        print('iter: %3d: %0.5f'%(iters, loss.item()))

    return pose.detach(), shape.detach(), verts_pred.squeeze().detach().cpu().numpy(), verts_gt.squeeze().cpu().numpy()


def infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges, thresh=-1e-2):
    with torch.no_grad():
        uv_faces_torch_f = torch.LongTensor(uv_faces).cuda()
        uv_faces_torch_b = torch.LongTensor(uv_faces[:,[0,2,1]]).cuda()
        vertices_new_f = uv_vertices[:,:2].clone()
        vertices_new_b = uv_vertices[:,:2].clone()

        uv_input = uv_vertices[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.unsqueeze(0).repeat(num_points, 1)

        pred_f = model_sdf_f(uv_input, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]
        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1)
        label_b = torch.argmax(label_b, dim=-1)

        sdf_pred = torch.stack((sdf_pred_f, sdf_pred_b), dim=0)
        uv_vertices_batch = torch.stack((uv_vertices[:,:2], uv_vertices[:,:2]), dim=0)
        vertices_new, faces_list = mesh_reader.read_mesh_from_sdf(uv_vertices_batch, uv_faces_torch_f, sdf_pred, edges, thresh=thresh)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]

        vertices_new_f, faces_new_f = mesh_reader.reorder_vertices_faces(vertices_new_f.cpu().numpy(), faces_new_f.cpu().numpy())
        vertices_new_b, faces_new_b = mesh_reader.reorder_vertices_faces(vertices_new_b.cpu().numpy(), faces_new_b.cpu().numpy())

        vertices_new_f = torch.FloatTensor(vertices_new_f).cuda()
        vertices_new_b = torch.FloatTensor(vertices_new_b).cuda()
        faces_new_f = torch.FloatTensor(faces_new_f).cuda()
        faces_new_b = torch.FloatTensor(faces_new_b).cuda()
        
    return vertices_new_f, vertices_new_b, faces_new_f, faces_new_b

def retrieve_cloth(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, packed_body, rotate_mat, save_folder, cloth_type=0):

    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()
    verts_body, faces_body, root_J = packed_body

    best_i = 0
    loss_best = 1e10
    latent_code_best = latent_codes[0].clone().detach()
    for i in range(200):
        latent_code = latent_codes[i].clone().detach()

        pattern_vertices_f, pattern_vertices_b, faces_f, faces_b = infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges)
        pattern_vertices_f = pattern_vertices_f*10
        pattern_vertices_b = pattern_vertices_b*10

        num_points_f = len(pattern_vertices_f)
        num_points_b = len(pattern_vertices_b)

        latent_code_input_f = latent_code.unsqueeze(0).repeat(num_points_f, 1)
        latent_code_input_b = latent_code.unsqueeze(0).repeat(num_points_b, 1)

        pattern_vertices_f_new = pattern_vertices_f.detach()
        pattern_vertices_b_new = pattern_vertices_b.detach()

        pred_atlas_f = model_atlas_f(pattern_vertices_f_new, latent_code_input_f.detach())/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b_new, latent_code_input_b.detach())/10

        pattern_vertices_f_3D = np.concatenate((pattern_vertices_f.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_f), 1))), axis=-1)
        pattern_vertices_b_3D = np.concatenate((pattern_vertices_b.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_b), 1))), axis=-1)
        mesh_pattern_f = trimesh.Trimesh(pattern_vertices_f_3D, faces_f.cpu().numpy(), validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(pattern_vertices_b_3D, faces_b.cpu().numpy(), validate=False, process=False)
        
        fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200)

        # for SMPL
        pred_atlas_f = pred_atlas_f*diag_max/2 
        pred_atlas_b = pred_atlas_b*diag_max/2 
        y_center_offset_f = torch.zeros_like(pred_atlas_f)
        y_center_offset_b = torch.zeros_like(pred_atlas_b)
        y_center_offset_f[:, 1] = y_center
        y_center_offset_b[:, 1] = y_center
        pred_atlas_f = pred_atlas_f + y_center_offset_f
        pred_atlas_b = pred_atlas_b + y_center_offset_b

        vertices_garment_T_f = pred_atlas_f
        vertices_garment_T_b = pred_atlas_b

        barycentric_uv_f = torch.FloatTensor(barycentric_uv_f).cuda()
        barycentric_uv_b = torch.FloatTensor(barycentric_uv_b).cuda()
        closest_face_idx_uv_f = torch.LongTensor(closest_face_idx_uv_f).cuda()
        closest_face_idx_uv_b = torch.LongTensor(closest_face_idx_uv_b).cuda()
        
        packed_input = [vertices_garment_T_f, vertices_garment_T_b, faces_f.cpu().numpy(), faces_b.cpu().numpy(), barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code]
        _, _, cloth_v_deformed, cloth_f_deformed = draping.draping(packed_input, pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl)
        cloth_v_deformed = cloth_v_deformed.squeeze()
        cloth_f_deformed = cloth_f_deformed.squeeze()

        verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed), 3)
        faces = torch.cat((faces_body, cloth_f_deformed + len(verts_body)))
        smpl_rgb = torch.zeros(len(verts_body), 3)
        smpl_rgb[:,0] += 255
        cloth_rgb = torch.zeros(len(cloth_v_deformed), 3)
        cloth_rgb[:,1] += 255
        verts_rgb = torch.cat((smpl_rgb, cloth_rgb))[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())

        mesh = Meshes(
            verts=[verts_zero.cuda()],   
            faces=[faces.cuda()],
            textures=textures
        )

        verts_deformed = torch.cat((verts_body, cloth_v_deformed), dim=0)
        verts_deformed = torch.einsum('ij,nj->ni', rotate_mat, verts_deformed - root_J) + root_J

        signs = torch.ones_like(verts_deformed).cuda()
        signs[:,:2] *= -1
        verts_deformed += (signs-1)*root_J
        
        new_src_mesh = mesh.offset_verts(verts_deformed)
        images_pred = renderer_textured(new_src_mesh)
        images_pred = images_pred[0, :, :, :3]/255

        cloth_silh = images_pred[:, :, 1]
        cloth_silh = torch.sigmoid(cloth_silh*10)*2 - 1
        cloth_gt = images_gt[:, :, cloth_type]
        
        intersection_g = (cloth_silh*cloth_gt).sum()
        union_g = cloth_silh.sum() + cloth_gt.sum() - intersection_g
        loss_texture = (1 - intersection_g/union_g)*224

        print('cloth %04d, loss_texture: %0.5f'%(i, loss_texture.item()))
        loss = loss_texture

        if loss_best > loss.item():
            best_i = i
            loss_best = loss.item()
            latent_code_best = latent_code.clone().detach()

            cv2.imwrite(save_folder + '/best-text.png', (images_pred.detach().cpu().numpy()*255).astype(np.uint8))

            faces_new = new_src_mesh._faces_packed.detach().cpu().numpy()
            verts_new = new_src_mesh._verts_packed.detach().cpu().numpy()
            mesh_new = trimesh.Trimesh(verts_new, faces_new)
            colors_f_b = np.ones((len(faces_body), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
            colors_f_g = np.ones((len(cloth_f_deformed), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
            colors_f = np.concatenate((colors_f_b, colors_f_g))
            mesh_new.visual.face_colors = colors_f
            mesh_new.export(save_folder + '/retrieve-best.obj')

    return best_i


def fitting(best_i, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, packed_body, rotate_mat, body, save_folder, cloth_type=0, thresh=-1e-2):
    
    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()
    verts_body, faces_body, root_J = packed_body
    body.update_body(verts_body.unsqueeze(0))

    lr = 1e-3
    iters = 50
    latent_code = latent_codes[best_i].clone().detach()
    latent_code.requires_grad = True 
    optimizer = torch.optim.Adam([{'params': latent_code, 'lr': lr}])

    loss_best = 1e10
    latent_code_best = latent_codes[best_i].clone().detach()
    latent_code_fake = latent_codes[best_i].clone().detach()
    for step in range(iters):

        pattern_vertices_f, pattern_vertices_b, faces_f, faces_b = infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges, thresh=thresh)
        pattern_vertices_f = pattern_vertices_f*10
        pattern_vertices_b = pattern_vertices_b*10
        pattern_vertices_f.requires_grad = True
        pattern_vertices_b.requires_grad = True

        num_points_f = len(pattern_vertices_f)
        num_points_b = len(pattern_vertices_b)

        latent_code_input_f = latent_code.unsqueeze(0).repeat(num_points_f, 1)
        latent_code_input_b = latent_code.unsqueeze(0).repeat(num_points_b, 1)
        pred_f = model_sdf_f(pattern_vertices_f, latent_code_input_f)
        pred_b = model_sdf_b(pattern_vertices_b, latent_code_input_b)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]

        sdf_pred_f.sum().backward(retain_graph=True)
        sdf_pred_b.sum().backward(retain_graph=True)

        normals_f = pattern_vertices_f.grad
        normals_b = pattern_vertices_b.grad
        normals_f = normals_f / (torch.norm(normals_f, dim=-1, keepdim=True) + 0.0001)
        normals_b = normals_b / (torch.norm(normals_b, dim=-1, keepdim=True) + 0.0001)

        ### to bring bck differentiability
        pattern_vertices_f_new = pattern_vertices_f.detach() - (sdf_pred_f - sdf_pred_f.detach())[:, None] * normals_f
        pattern_vertices_b_new = pattern_vertices_b.detach() - (sdf_pred_b - sdf_pred_b.detach())[:, None] * normals_b

        pred_atlas_f = model_atlas_f(pattern_vertices_f_new, latent_code_input_f.detach())/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b_new, latent_code_input_b.detach())/10

        pattern_vertices_f_3D = np.concatenate((pattern_vertices_f.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_f), 1))), axis=-1)
        pattern_vertices_b_3D = np.concatenate((pattern_vertices_b.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_b), 1))), axis=-1)
        mesh_pattern_f = trimesh.Trimesh(pattern_vertices_f_3D, faces_f.cpu().numpy(), validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(pattern_vertices_b_3D, faces_b.cpu().numpy(), validate=False, process=False)

        fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200)

        # for SMPL
        pred_atlas_f = pred_atlas_f*diag_max/2 
        pred_atlas_b = pred_atlas_b*diag_max/2 
        y_center_offset_f = torch.zeros_like(pred_atlas_f)
        y_center_offset_b = torch.zeros_like(pred_atlas_b)
        y_center_offset_f[:, 1] = y_center
        y_center_offset_b[:, 1] = y_center
        pred_atlas_f = pred_atlas_f + y_center_offset_f
        pred_atlas_b = pred_atlas_b + y_center_offset_b

        vertices_garment_T_f = pred_atlas_f
        vertices_garment_T_b = pred_atlas_b

        barycentric_uv_f = torch.FloatTensor(barycentric_uv_f).cuda()
        barycentric_uv_b = torch.FloatTensor(barycentric_uv_b).cuda()
        closest_face_idx_uv_f = torch.LongTensor(closest_face_idx_uv_f).cuda()
        closest_face_idx_uv_b = torch.LongTensor(closest_face_idx_uv_b).cuda()

        packed_input = [vertices_garment_T_f, vertices_garment_T_b, faces_f.cpu().numpy(), faces_b.cpu().numpy(), barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code.detach()]
        _, _, cloth_v_deformed, cloth_f_deformed = draping.draping_grad(packed_input, pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl)
        cloth_v_deformed = cloth_v_deformed.squeeze()
        cloth_f_deformed = cloth_f_deformed.squeeze()

        verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed), 3)
        faces = torch.cat((faces_body, cloth_f_deformed.cpu() + len(verts_body)))
        smpl_rgb = torch.zeros(len(verts_body), 3)
        smpl_rgb[:,0] += 255
        cloth_rgb = torch.zeros(len(cloth_v_deformed), 3)
        cloth_rgb[:,1] += 255
        verts_rgb = torch.cat((smpl_rgb, cloth_rgb))[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())

        mesh = Meshes(
            verts=[verts_zero.cuda()],   
            faces=[faces.cuda()],
            textures=textures
        )

        verts_deformed = torch.cat((verts_body, cloth_v_deformed), dim=0)
        verts_deformed = torch.einsum('ij,nj->ni', rotate_mat, verts_deformed - root_J) + root_J
        signs = torch.ones_like(verts_deformed).cuda()
        signs[:,:2] *= -1
        verts_deformed += (signs-1)*root_J

        new_src_mesh = mesh.offset_verts(verts_deformed)
        images_pred = renderer_textured(new_src_mesh)
        images_pred = images_pred[0, :, :, :3]/255

        cloth_silh = images_pred[:, :, 1]
        #cloth_silh = torch.sigmoid(cloth_silh*10)*2 - 1
        cloth_gt = images_gt[:, :, cloth_type]
        
        intersection_g = (cloth_silh*cloth_gt).sum()
        union_g = cloth_silh.sum() + cloth_gt.sum() - intersection_g
        loss_texture = (1 - intersection_g/union_g)*224

        loss_rep = latent_code.norm(dim=-1)/2

        loss_collision = collision_penalty(cloth_v_deformed.unsqueeze(0), body.vb, body.nb, eps=5e-3, kcollision=250)

        print('step %04d, loss_texture: %0.5f, Loss_rep: %0.5f, Loss_collision: %0.5f'%(step, loss_texture.item(), loss_rep.item(), loss_collision.item()))
        loss = loss_texture + loss_rep + loss_collision

        if loss_best > loss.item():
            loss_best = loss.item()
            latent_code_best = latent_code.clone().detach()
            torch.save(latent_code_best.cpu().detach(), save_folder + '/best-z-fit-%d.pt'%cloth_type)

            cv2.imwrite(save_folder + '/best-z-fit-%d-text.png'%cloth_type, (images_pred.detach().cpu().numpy()*255).astype(np.uint8))

            faces_new = new_src_mesh._faces_packed.detach().cpu().numpy()
            verts_new = new_src_mesh._verts_packed.detach().cpu().numpy()
            mesh_new = trimesh.Trimesh(verts_new, faces_new)
            colors_f_b = np.ones((len(faces_body), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
            colors_f_g = np.ones((len(cloth_f_deformed), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
            colors_f = np.concatenate((colors_f_b, colors_f_g))
            mesh_new.visual.face_colors = colors_f
            trimesh.smoothing.filter_taubin(mesh_new, lamb=0.5)
            mesh_new.export(save_folder + '/best-z-fit-%d-clothed.obj'%cloth_type)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return latent_code_best.detach()


def prepare_draping_top(latent_code_top, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, uv_vertices, uv_faces, edges, mesh_uv_200, diag_max, y_center, pose, beta, model_diffusion, model_draping, packed_input_smpl, resolution=200):
    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()

    packed_input = []
    packed_input_atlas = []
    faces_sewing = []
    packed_skinning = []
    for i in range(len(latent_code_top)):
        print('Draping ', i)

        latent_code = latent_code_top[i]

        mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = ISP.reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code, uv_vertices, uv_faces, edges, resolution=resolution, which='tee')

        fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

        faces_sewing.append(mesh_sewing.faces)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200, return_tensor=True)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200, return_tensor=True)

        barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b = draping.prepare_barycentric_uv2atlas(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, mesh_uv_200, res=200, return_tensor=True)

        packed_input_atlas_i = [barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b]

        # for SMPL
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

        
        garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = draping.draping(packed_input[i], pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl)
        packed_skinning_i = [garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces]

        packed_skinning.append(packed_skinning_i)

    packed_top = [packed_input, packed_input_atlas, faces_sewing, packed_skinning]

    return packed_top


def retrieve_cloth_layering(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=0, is_pants=False, Rot_rest=None, pose_offsets_rest=None):

    packed_input, packed_input_atlas, faces_sewing, packed_skinning = packed_top
    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()
    verts_body, faces_body, root_J = packed_body
    body.update_body(verts_body.unsqueeze(0))

    best_i = 0
    loss_best = 1e10
    latent_code_best = latent_codes[0].clone().detach()
    for i in range(200):
        latent_code = latent_codes[i].clone().detach()

        pattern_vertices_f, pattern_vertices_b, faces_f, faces_b = infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges)
        pattern_vertices_f = pattern_vertices_f*10
        pattern_vertices_b = pattern_vertices_b*10

        num_points_f = len(pattern_vertices_f)
        num_points_b = len(pattern_vertices_b)

        latent_code_input_f = latent_code.unsqueeze(0).repeat(num_points_f, 1)
        latent_code_input_b = latent_code.unsqueeze(0).repeat(num_points_b, 1)

        pattern_vertices_f_new = pattern_vertices_f.detach()
        pattern_vertices_b_new = pattern_vertices_b.detach()

        pred_atlas_f = model_atlas_f(pattern_vertices_f_new, latent_code_input_f.detach())/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b_new, latent_code_input_b.detach())/10

        pattern_vertices_f_3D = np.concatenate((pattern_vertices_f.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_f), 1))), axis=-1)
        pattern_vertices_b_3D = np.concatenate((pattern_vertices_b.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_b), 1))), axis=-1)
        mesh_pattern_f = trimesh.Trimesh(pattern_vertices_f_3D, faces_f.cpu().numpy(), validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(pattern_vertices_b_3D, faces_b.cpu().numpy(), validate=False, process=False)
        
        if is_pants:
            fix_mask = draping.generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        else:
            fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200)

        # for SMPL
        pred_atlas_f = pred_atlas_f*diag_max/2 
        pred_atlas_b = pred_atlas_b*diag_max/2 
        y_center_offset_f = torch.zeros_like(pred_atlas_f)
        y_center_offset_b = torch.zeros_like(pred_atlas_b)
        y_center_offset_f[:, 1] = y_center
        y_center_offset_b[:, 1] = y_center
        pred_atlas_f = pred_atlas_f + y_center_offset_f
        pred_atlas_b = pred_atlas_b + y_center_offset_b

        vertices_garment_T_f = pred_atlas_f
        vertices_garment_T_b = pred_atlas_b

        barycentric_uv_f = torch.FloatTensor(barycentric_uv_f).cuda()
        barycentric_uv_b = torch.FloatTensor(barycentric_uv_b).cuda()
        closest_face_idx_uv_f = torch.LongTensor(closest_face_idx_uv_f).cuda()
        closest_face_idx_uv_b = torch.LongTensor(closest_face_idx_uv_b).cuda()
        
        packed_input_u = [vertices_garment_T_f, vertices_garment_T_b, faces_f.cpu().numpy(), faces_b.cpu().numpy(), barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code]
        deformed_f, deformed_b, deformed, deformed_faces = draping.draping(packed_input_u, pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl, is_pants=is_pants, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)

        packed_skinning_u = [deformed_f.detach(), deformed_b.detach(), deformed.detach(), deformed_faces.detach()]

        packed_input_with_u = [packed_input_u] + packed_input[::-1]
        packed_input_atlas_with_u = [[]] + packed_input_atlas[::-1]
        packed_skinning_with_u = [packed_skinning_u] + packed_skinning[::-1]

        for u in range(len(packed_input_with_u)):
            is_bottom = u==0 and is_pants
            for j in range(u+1, len(packed_input_with_u)):
                garment_layer_f_j, garment_layer_b_j = layering.draping_layer(packed_input_with_u[u], packed_input_with_u[j], packed_input_atlas_with_u[j], packed_skinning_with_u[u], packed_skinning_with_u[j], pose, beta, model_diffusion, model_draping, model_layer, uv_faces_200, deformed_cloth, body=body, is_bottom=is_bottom)

                garment_skinning_f_j = garment_layer_f_j
                garment_skinning_b_j = garment_layer_b_j
                garment_skinning_j = torch.cat((garment_layer_f_j, garment_layer_b_j), dim=1)
                packed_skinning_with_u[j] = [garment_skinning_f_j, garment_skinning_b_j, garment_skinning_j, packed_skinning_with_u[j][-1]]

        cloth_v_deformed_u = deformed.squeeze()
        cloth_f_deformed_u = deformed_faces.squeeze()
        
        if is_pants:
            cloth_v_deformed_o_1 = packed_skinning_with_u[1][2].squeeze()
            cloth_f_deformed_o_1 = torch.LongTensor(faces_sewing[-1])
            cloth_v_deformed_o_2 = packed_skinning_with_u[2][2].squeeze()
            cloth_f_deformed_o_2 = torch.LongTensor(faces_sewing[-2])
            verts_deformed = torch.cat((verts_body, cloth_v_deformed_u, cloth_v_deformed_o_1, cloth_v_deformed_o_2), dim=0)

            verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o_1)+len(cloth_v_deformed_o_2), 3)
            faces = torch.cat((faces_body, cloth_f_deformed_u.cpu() + len(verts_body), cloth_f_deformed_o_1.cpu() + len(verts_body)+len(cloth_v_deformed_u), cloth_f_deformed_o_2.cpu() + len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o_1)))
            smpl_rgb = torch.zeros(len(verts_body), 3)
            smpl_rgb[:,0] += 255
            cloth_rgb_u = torch.zeros(len(cloth_v_deformed_u), 3)
            cloth_rgb_u[:,1] += 255
            cloth_rgb_o = torch.zeros(len(cloth_v_deformed_o_1)+len(cloth_v_deformed_o_2), 3)
            cloth_rgb_o[:,2] += 255
            verts_rgb = torch.cat((smpl_rgb, cloth_rgb_u, cloth_rgb_o))[None]
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
        else:
            cloth_v_deformed_o = packed_skinning_with_u[1][2].squeeze()
            cloth_f_deformed_o = torch.LongTensor(faces_sewing[0])
            verts_deformed = torch.cat((verts_body, cloth_v_deformed_u, cloth_v_deformed_o), dim=0)

            verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o), 3)
            faces = torch.cat((faces_body, cloth_f_deformed_u.cpu() + len(verts_body), cloth_f_deformed_o.cpu() + len(verts_body)+len(cloth_v_deformed_u)))
            smpl_rgb = torch.zeros(len(verts_body), 3)
            smpl_rgb[:,0] += 255
            cloth_rgb_u = torch.zeros(len(cloth_v_deformed_u), 3)
            cloth_rgb_u[:,1] += 255
            cloth_rgb_o = torch.zeros(len(cloth_v_deformed_o), 3)
            cloth_rgb_o[:,2] += 255
            verts_rgb = torch.cat((smpl_rgb, cloth_rgb_u, cloth_rgb_o))[None]
            textures = TexturesVertex(verts_features=verts_rgb.cuda())

        mesh = Meshes(
            verts=[verts_zero.cuda()],   
            faces=[faces.cuda()],
            textures=textures
        )

        verts_deformed = torch.einsum('ij,nj->ni', rotate_mat, verts_deformed - root_J) + root_J
        signs = torch.ones_like(verts_deformed).cuda()
        signs[:,:2] *= -1
        verts_deformed += (signs-1)*root_J

        new_src_mesh = mesh.offset_verts(verts_deformed)
        images_pred = renderer_textured(new_src_mesh)
        images_pred = images_pred[0, :, :, :3]/255

        cloth_silh = images_pred[:, :, 1]
        cloth_silh = torch.sigmoid(cloth_silh*10)*2 - 1
        cloth_gt = images_gt[:, :, cloth_type]
        
        intersection_g = (cloth_silh*cloth_gt).sum()
        union_g = cloth_silh.sum() + cloth_gt.sum() - intersection_g
        loss_texture = (1 - intersection_g/union_g)*224

        print('cloth %04d, loss_texture: %0.5f'%(i, loss_texture.item()))
        loss = loss_texture

        if loss_best > loss.item():
            best_i = i
            loss_best = loss.item()
            latent_code_best = latent_code.clone().detach()

            cv2.imwrite(save_folder + '/best-text.png', (images_pred.detach().cpu().numpy()*255).astype(np.uint8))

            faces_new = new_src_mesh._faces_packed.detach().cpu().numpy()
            verts_new = new_src_mesh._verts_packed.detach().cpu().numpy()
            mesh_new = trimesh.Trimesh(verts_new, faces_new)
            colors_f_b = np.ones((len(faces_body), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
            if is_pants:
                colors_f_g_u = np.ones((len(cloth_f_deformed_u), 4))*np.array([100, 100, 100, 200])[np.newaxis,:]
                colors_f_g_o_1 = np.ones((len(cloth_f_deformed_o_1), 4))*np.array([255, 160, 255, 200])[np.newaxis,:]
                colors_f_g_o_2 = np.ones((len(cloth_f_deformed_o_2), 4))*np.array([255, 255, 160, 200])[np.newaxis,:]
                colors_f = np.concatenate((colors_f_b, colors_f_g_u, colors_f_g_o_1, colors_f_g_o_2))
            else:
                colors_f_g_u = np.ones((len(cloth_f_deformed_u), 4))*np.array([255, 160, 255, 200])[np.newaxis,:]
                colors_f_g_o = np.ones((len(cloth_f_deformed_o), 4))*np.array([255, 255, 160, 200])[np.newaxis,:]
                colors_f = np.concatenate((colors_f_b, colors_f_g_u, colors_f_g_o))

            mesh_new.visual.face_colors = colors_f
            mesh_new.export(save_folder + '/retrieve-best.obj')

    return best_i


def fitting_layering(best_i, model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, edges, mesh_uv_200, renderer_textured, images_gt, diag_max, y_center, pose, beta, model_diffusion, model_draping, model_layer, packed_input_smpl, packed_body, packed_top, rotate_mat, body, deformed_cloth, save_folder, cloth_type=0, is_pants=False, Rot_rest=None, pose_offsets_rest=None):

    packed_input, packed_input_atlas, faces_sewing, packed_skinning = packed_top
    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()
    verts_body, faces_body, root_J = packed_body
    body.update_body(verts_body.unsqueeze(0))

    lr = 1e-3
    iters = 50
    
    latent_code = latent_codes[best_i].clone().detach()
    latent_code.requires_grad = True 
    optimizer = torch.optim.Adam([{'params': latent_code, 'lr': lr}])

    loss_best = 1e10
    latent_code_best = latent_codes[best_i].clone().detach()
    latent_code_fake = latent_codes[best_i].clone().detach()
    for step in range(iters):

        pattern_vertices_f, pattern_vertices_b, faces_f, faces_b = infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges)
        pattern_vertices_f = pattern_vertices_f*10
        pattern_vertices_b = pattern_vertices_b*10
        pattern_vertices_f.requires_grad = True
        pattern_vertices_b.requires_grad = True

        num_points_f = len(pattern_vertices_f)
        num_points_b = len(pattern_vertices_b)

        latent_code_input_f = latent_code.unsqueeze(0).repeat(num_points_f, 1)
        latent_code_input_b = latent_code.unsqueeze(0).repeat(num_points_b, 1)
        pred_f = model_sdf_f(pattern_vertices_f, latent_code_input_f)
        pred_b = model_sdf_b(pattern_vertices_b, latent_code_input_b)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]
        sdf_pred_f.sum().backward(retain_graph=True)
        sdf_pred_b.sum().backward(retain_graph=True)

        normals_f = pattern_vertices_f.grad
        normals_b = pattern_vertices_b.grad
        normals_f = normals_f / (torch.norm(normals_f, dim=-1, keepdim=True) + 0.0001)
        normals_b = normals_b / (torch.norm(normals_b, dim=-1, keepdim=True) + 0.0001)

        ### to bring bck differentiability
        pattern_vertices_f_new = pattern_vertices_f.detach() - (sdf_pred_f - sdf_pred_f.detach())[:, None] * normals_f
        pattern_vertices_b_new = pattern_vertices_b.detach() - (sdf_pred_b - sdf_pred_b.detach())[:, None] * normals_b

        pred_atlas_f = model_atlas_f(pattern_vertices_f_new, latent_code_input_f.detach())/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b_new, latent_code_input_b.detach())/10

        pattern_vertices_f_3D = np.concatenate((pattern_vertices_f.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_f), 1))), axis=-1)
        pattern_vertices_b_3D = np.concatenate((pattern_vertices_b.detach().cpu().numpy()/10, np.zeros((len(pattern_vertices_b), 1))), axis=-1)
        mesh_pattern_f = trimesh.Trimesh(pattern_vertices_f_3D, faces_f.cpu().numpy(), validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(pattern_vertices_b_3D, faces_b.cpu().numpy(), validate=False, process=False)
        
        if is_pants:
            fix_mask = draping.generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        else:
            fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200)

        # for SMPL
        pred_atlas_f = pred_atlas_f*diag_max/2 
        pred_atlas_b = pred_atlas_b*diag_max/2 
        y_center_offset_f = torch.zeros_like(pred_atlas_f)
        y_center_offset_b = torch.zeros_like(pred_atlas_b)
        y_center_offset_f[:, 1] = y_center
        y_center_offset_b[:, 1] = y_center
        pred_atlas_f = pred_atlas_f + y_center_offset_f
        pred_atlas_b = pred_atlas_b + y_center_offset_b

        vertices_garment_T_f = pred_atlas_f
        vertices_garment_T_b = pred_atlas_b

        barycentric_uv_f = torch.FloatTensor(barycentric_uv_f).cuda()
        barycentric_uv_b = torch.FloatTensor(barycentric_uv_b).cuda()
        closest_face_idx_uv_f = torch.LongTensor(closest_face_idx_uv_f).cuda()
        closest_face_idx_uv_b = torch.LongTensor(closest_face_idx_uv_b).cuda()
        
        packed_input_u = [vertices_garment_T_f, vertices_garment_T_b, faces_f.cpu().numpy(), faces_b.cpu().numpy(), barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask, latent_code.detach()]
        deformed_f, deformed_b, deformed, deformed_faces = draping.draping_grad(packed_input_u, pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl, is_pants=is_pants, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)

        packed_skinning_u = [deformed_f.detach(), deformed_b.detach(), deformed.detach(), deformed_faces.detach()]

        packed_input_with_u = [packed_input_u] + packed_input[::-1]
        packed_input_atlas_with_u = [[]] + packed_input_atlas[::-1]
        packed_skinning_with_u = [packed_skinning_u] + packed_skinning[::-1]

        for u in range(len(packed_input_with_u)):
            is_bottom = u==0 and is_pants
            for j in range(u+1, len(packed_input_with_u)):
                garment_layer_f_j, garment_layer_b_j = layering.draping_layer(packed_input_with_u[u], packed_input_with_u[j], packed_input_atlas_with_u[j], packed_skinning_with_u[u], packed_skinning_with_u[j], pose, beta, model_diffusion, model_draping, model_layer, uv_faces_200, deformed_cloth, body=body, is_bottom=is_bottom)

                garment_skinning_f_j = garment_layer_f_j
                garment_skinning_b_j = garment_layer_b_j
                garment_skinning_j = torch.cat((garment_layer_f_j, garment_layer_b_j), dim=1)
                packed_skinning_with_u[j] = [garment_skinning_f_j, garment_skinning_b_j, garment_skinning_j, packed_skinning_with_u[j][-1]]

        cloth_v_deformed_u = deformed.squeeze()
        cloth_f_deformed_u = deformed_faces.squeeze()
        
        if is_pants:
            cloth_v_deformed_o_1 = packed_skinning_with_u[1][2].squeeze()
            cloth_f_deformed_o_1 = torch.LongTensor(faces_sewing[-1])
            cloth_v_deformed_o_2 = packed_skinning_with_u[2][2].squeeze()
            cloth_f_deformed_o_2 = torch.LongTensor(faces_sewing[-2])
            verts_deformed = torch.cat((verts_body, cloth_v_deformed_u, cloth_v_deformed_o_1, cloth_v_deformed_o_2), dim=0)

            verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o_1)+len(cloth_v_deformed_o_2), 3)
            faces = torch.cat((faces_body, cloth_f_deformed_u.cpu() + len(verts_body), cloth_f_deformed_o_1.cpu() + len(verts_body)+len(cloth_v_deformed_u), cloth_f_deformed_o_2.cpu() + len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o_1)))
            smpl_rgb = torch.zeros(len(verts_body), 3)
            smpl_rgb[:,0] += 255
            cloth_rgb_u = torch.zeros(len(cloth_v_deformed_u), 3)
            cloth_rgb_u[:,1] += 255
            cloth_rgb_o = torch.zeros(len(cloth_v_deformed_o_1)+len(cloth_v_deformed_o_2), 3)
            cloth_rgb_o[:,2] += 255
            verts_rgb = torch.cat((smpl_rgb, cloth_rgb_u, cloth_rgb_o))[None]
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
        else:
            cloth_v_deformed_o = packed_skinning_with_u[1][2].squeeze()
            cloth_f_deformed_o = torch.LongTensor(faces_sewing[0])
            verts_deformed = torch.cat((verts_body, cloth_v_deformed_u, cloth_v_deformed_o), dim=0)

            verts_zero = torch.zeros(len(verts_body)+len(cloth_v_deformed_u)+len(cloth_v_deformed_o), 3)
            faces = torch.cat((faces_body, cloth_f_deformed_u.cpu() + len(verts_body), cloth_f_deformed_o.cpu() + len(verts_body)+len(cloth_v_deformed_u)))
            smpl_rgb = torch.zeros(len(verts_body), 3)
            smpl_rgb[:,0] += 255
            cloth_rgb_u = torch.zeros(len(cloth_v_deformed_u), 3)
            cloth_rgb_u[:,1] += 255
            cloth_rgb_o = torch.zeros(len(cloth_v_deformed_o), 3)
            cloth_rgb_o[:,2] += 255
            verts_rgb = torch.cat((smpl_rgb, cloth_rgb_u, cloth_rgb_o))[None]
            textures = TexturesVertex(verts_features=verts_rgb.cuda())

        mesh = Meshes(
            verts=[verts_zero.cuda()],   
            faces=[faces.cuda()],
            textures=textures
        )

        verts_deformed = torch.einsum('ij,nj->ni', rotate_mat, verts_deformed - root_J) + root_J
        signs = torch.ones_like(verts_deformed).cuda()
        signs[:,:2] *= -1
        verts_deformed += (signs-1)*root_J

        new_src_mesh = mesh.offset_verts(verts_deformed)
        images_pred = renderer_textured(new_src_mesh)
        images_pred = images_pred[0, :, :, :3]/255

        cloth_silh = images_pred[:, :, 1]
        cloth_silh = torch.sigmoid(cloth_silh*10)*2 - 1
        cloth_gt = images_gt[:, :, cloth_type]
        
        intersection_g = (cloth_silh*cloth_gt).sum()
        union_g = cloth_silh.sum() + cloth_gt.sum() - intersection_g
        loss_texture = (1 - intersection_g/union_g)*224
        loss_collision = collision_penalty(cloth_v_deformed_u.unsqueeze(0), body.vb, body.nb, eps=5e-3, kcollision=250)

        loss_rep = latent_code.norm(dim=-1)/2/5
        print('step %04d, loss_texture: %0.5f, Loss_rep: %0.5f, Loss_collision: %0.5f'%(step, loss_texture.item(), loss_rep.item(), loss_collision.item()))
        loss = loss_texture + loss_rep + loss_collision

        if loss_best > loss.item():
            loss_best = loss.item()
            latent_code_best = latent_code.clone().detach()
            torch.save(latent_code_best.cpu().detach(), save_folder + '/best-z-fit-%d.pt'%cloth_type)

            cv2.imwrite(save_folder + '/best-z-fit-%d-text.png'%cloth_type, (images_pred.detach().cpu().numpy()*255).astype(np.uint8))

            faces_new = new_src_mesh._faces_packed.detach().cpu().numpy()
            verts_new = new_src_mesh._verts_packed.detach().cpu().numpy()
            mesh_new = trimesh.Trimesh(verts_new, faces_new)
            colors_f_b = np.ones((len(faces_body), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
            if is_pants:
                colors_f_g_u = np.ones((len(cloth_f_deformed_u), 4))*np.array([100, 100, 100, 200])[np.newaxis,:]
                colors_f_g_o_1 = np.ones((len(cloth_f_deformed_o_1), 4))*np.array([255, 160, 255, 200])[np.newaxis,:]
                colors_f_g_o_2 = np.ones((len(cloth_f_deformed_o_2), 4))*np.array([255, 255, 160, 200])[np.newaxis,:]
                colors_f = np.concatenate((colors_f_b, colors_f_g_u, colors_f_g_o_1, colors_f_g_o_2))
            else:
                colors_f_g_u = np.ones((len(cloth_f_deformed_u), 4))*np.array([255, 160, 255, 200])[np.newaxis,:]
                colors_f_g_o = np.ones((len(cloth_f_deformed_o), 4))*np.array([255, 255, 160, 200])[np.newaxis,:]
                colors_f = np.concatenate((colors_f_b, colors_f_g_u, colors_f_g_o))

            mesh_new.visual.face_colors = colors_f
            mesh_new.export(save_folder + '/best-z-fit-%d-clothed.obj'%cloth_type)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return latent_code_best.detach()


def infer(latent_codes, top, bottom, model_diffusion, model_diffusion_TA, model_layer, uv_vertices, uv_faces, edges, mesh_uv_200, packed_input_smpl, packed_body, rotate_mat, pose, beta, body, deformed_cloth, save_folder, resolution=200, is_pants=True, Rot_rest=None, pose_offsets_rest=None):

    uv_faces_200 = torch.LongTensor(mesh_uv_200.faces).cuda()
    verts_body, faces_body, root_J = packed_body
    root_J = root_J.cpu().numpy()
    body.update_body(verts_body.unsqueeze(0))

    rotate_mat = rotate_mat.cpu().numpy()

    model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, model_draping, diag_max, y_center = top
    model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, model_draping_bottom, diag_max_bottom, y_center_bottom = bottom


    packed_input = []
    packed_input_atlas = []
    faces_sewing = []
    for i in range(len(latent_codes)):
        print('Reconstruct ', i)
        latent_code = latent_codes[i]
        if i == 0:
            which = 'pants' if is_pants else 'skirt'

            mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = ISP.reconstruct_batch(model_sdf_f_bottom, model_sdf_b_bottom, model_atlas_f_bottom, model_atlas_b_bottom, latent_code, uv_vertices, uv_faces, edges, resolution=resolution, which=which)

            fix_mask = draping.generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

        else:
            mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = ISP.reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code, uv_vertices, uv_faces, edges, resolution=resolution, which='tee')

            fix_mask = draping.generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

        faces_sewing.append(mesh_sewing.faces)
        fix_mask = torch.FloatTensor(fix_mask).cuda()

        barycentric_uv_f, closest_face_idx_uv_f = draping.barycentric_faces(mesh_pattern_f, mesh_uv_200, return_tensor=True)
        barycentric_uv_b, closest_face_idx_uv_b = draping.barycentric_faces(mesh_pattern_b, mesh_uv_200, return_tensor=True)

        if i != 0:
            barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b = draping.prepare_barycentric_uv2atlas(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, mesh_uv_200, res=200, return_tensor=True)

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
    for i in range(len(latent_codes)):
        if i == 0:
            model_diffusion_which = model_diffusion_TA if is_pants else model_diffusion
            garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = draping.draping(packed_input[i], pose, beta, model_diffusion_which, model_draping_bottom, uv_faces_200, packed_input_smpl, is_pants=is_pants, Rot_rest=Rot_rest, pose_offsets_rest=pose_offsets_rest)
        else:
            garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = draping.draping(packed_input[i], pose, beta, model_diffusion, model_draping, uv_faces_200, packed_input_smpl)
        packed_skinning_i = [garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces]

        packed_skinning.append(packed_skinning_i)


    ############# layering #############
    for i in range(len(latent_codes)):
        is_bottom = i==0
        for j in range(i+1, len(latent_codes)):
            print('Layering ', i, j)
            
            
            garment_layer_f_j, garment_layer_b_j = layering.draping_layer(packed_input[i], packed_input[j], packed_input_atlas[j], packed_skinning[i], packed_skinning[j], pose, beta, model_diffusion, model_draping, model_layer, uv_faces_200, deformed_cloth, body=body, is_bottom=is_bottom)

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

    for i in range(len(latent_codes)):
        print('Saving ', i)
        garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = packed_skinning[i]
        mesh_layer = trimesh.Trimesh(garment_skinning.squeeze().cpu().numpy(), faces_sewing[i], process=False, validate=False)

        mesh_layer.vertices = np.einsum('ij,nj->ni', rotate_mat, mesh_layer.vertices - root_J) + root_J

        #if i<2:
        #    mesh_layer = resolve_collision(mesh_layer, mesh_body, scale=2)

        colors_v = np.ones((len(mesh_layer.vertices), 3))*color_map[i][np.newaxis,:]
        mesh_layer.visual.vertex_colors = colors_v
        trimesh.smoothing.filter_taubin(mesh_layer, lamb=0.5)

        mesh_gar_layer_name = 'layer_bottom_%03d.obj'%i
        mesh_layer.export(os.path.join(save_folder, mesh_gar_layer_name), include_color=True)

    mesh_body = trimesh.Trimesh(verts_body.squeeze().detach().cpu().numpy(), faces_body.detach().cpu().numpy())
    mesh_body.vertices = np.einsum('ij,nj->ni', rotate_mat, mesh_body.vertices - root_J) + root_J
    
    mesh_body.export(os.path.join(save_folder, 'body.obj'))

    return

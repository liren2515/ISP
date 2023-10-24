import os,sys
import numpy as np 
import torch
import trimesh
import random
import argparse

from networks import SDF
from utils.ISP import reconstruct_batch
from utils import mesh_reader
from pytorch3d.ops.knn import knn_points


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def load_model(which, data_path='./extra-data', ckpt_path='./checkpoints'):
    rep_size = 32
    model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()

    if which == 'tee':
        statistic = np.load(os.path.join(data_path, 'shirt.npz'))
        y_center = statistic['y_center']
        diag_max = statistic['diag_max']

        model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+11, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+10, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_rep = SDF.learnt_representations(rep_size=rep_size, samples=400).cuda()

        model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_sdf_f.pth')))
        model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_sdf_b.pth')))
        model_rep.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_rep.pth')))

        model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_atlas_f.pth')))
        model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'shirt_atlas_b.pth')))

        gt_mesh = os.path.join(data_path, 'meshes/tee-gt.obj')
    elif which == 'pants':
        statistic = np.load(os.path.join(data_path, 'pants.npz'))
        y_center = statistic['y_center']
        diag_max = statistic['diag_max']

        model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+7, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+7, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_rep = SDF.learnt_representations(rep_size=rep_size, samples=200).cuda()

        model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_sdf_f.pth')))
        model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_sdf_b.pth')))
        model_rep.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_rep.pth')))

        model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_atlas_f.pth')))
        model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'pants_atlas_b.pth')))

        gt_mesh = os.path.join(data_path, 'meshes/pants-gt.obj')
    elif which == 'skirt':
        statistic = np.load(os.path.join(data_path, 'skirt.npz'))
        y_center = statistic['y_center']
        diag_max = statistic['diag_max']

        model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+4, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+4, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
        model_rep = SDF.learnt_representations(rep_size=rep_size, samples=300).cuda()
        
        model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_sdf_f.pth')))
        model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_sdf_b.pth')))
        model_rep.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_rep.pth')))

        model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_atlas_f.pth')))
        model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'skirt_atlas_b.pth')))

        gt_mesh = os.path.join(data_path, 'meshes/skirt-gt.obj')
    
    gt_mesh = trimesh.load(gt_mesh)

    return model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, diag_max, y_center, gt_mesh


def infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges):
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
        vertices_new, faces_list = mesh_reader.read_mesh_from_sdf(uv_vertices_batch, uv_faces_torch_f, sdf_pred, edges)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]
        unique_v_f = torch.unique(faces_new_f.reshape(-1))
        unique_v_b = torch.unique(faces_new_b.reshape(-1))


        pattern_vertices_f = vertices_new_f[unique_v_f][:,:2]
        pattern_vertices_b = vertices_new_b[unique_v_b][:,:2]

    return pattern_vertices_f.detach(), pattern_vertices_b.detach()

def fitting(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, target_mesh, 
    iters=1000):

    target_vertices = torch.FloatTensor(target_mesh.vertices[np.unique(target_mesh.faces.flatten())]).cuda().unsqueeze(0)
    uv_faces_torch_f = torch.LongTensor(uv_faces).cuda()
    uv_faces_torch_b = torch.LongTensor(uv_faces[:,[0,2,1]]).cuda()
    vertices_new_f = uv_vertices[:,:2].clone()
    vertices_new_b = uv_vertices[:,:2].clone()

    lr = 1e-3
    latent_code = latent_codes.mean(dim=0)
    latent_code.requires_grad = True 
    optimizer = torch.optim.Adam([{'params': latent_code, 'lr': lr}])
    
    uv_input = uv_vertices[:,:2]*10
    num_points = len(uv_vertices)
    y = target_vertices

    loss_best = 1e10
    latent_code_best = latent_codes.mean(dim=0).clone().detach()
    for step in range(iters):

        pattern_vertices_f, pattern_vertices_b = infer_vertices(model_sdf_f, model_sdf_b, latent_code, uv_vertices, uv_faces, edges)
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

        ### to bring back differentiability
        pattern_vertices_f_new = pattern_vertices_f.detach() - (sdf_pred_f - sdf_pred_f.detach())[:, None] * normals_f
        pattern_vertices_b_new = pattern_vertices_b.detach() - (sdf_pred_b - sdf_pred_b.detach())[:, None] * normals_b

        pred_atlas_f = model_atlas_f(pattern_vertices_f_new, latent_code_input_f)/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b_new, latent_code_input_b)/10

        x = torch.cat((pred_atlas_f, pred_atlas_b), dim=0)

        x = x.unsqueeze(0)
        x_nn = knn_points(x, y, K=1)
        y_nn = knn_points(y, x, K=1)

        cham_x = x_nn.dists.squeeze()  # (N, P1)
        cham_y = y_nn.dists.squeeze()  # (N, P2)

        cham_dist = cham_x.mean() + cham_y.mean()
        print('step:%04d, loss: %0.6f'%(step, cham_dist.item()))
        
        if loss_best>cham_dist.item():
            loss_best = cham_dist.item()
            latent_code_best = latent_code.clone().detach()

        optimizer.zero_grad()
        cham_dist.backward()
        optimizer.step()

    return latent_code_best.detach()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--which', default="tee", type=str, help="tee/pants/skirt")
    parser.add_argument('--res', default=256, type=int, help="resolution")
    parser.add_argument('--save_path', type=str, default='./tmp')
    parser.add_argument('--save_name', type=str, default='tee-fit')

    args = parser.parse_args()

    which = args.which
    save_path = args.save_path
    save_name = args.save_name
    x_res = y_res = args.res

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, diag_max, y_center, gt_mesh = load_model(which)

    uv_vertices, uv_faces = mesh_reader.create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
    edges = torch.LongTensor(mesh_uv.edges).cuda()
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()

    latent_codes = model_rep.weights.detach()

    gt_mesh_rescale = trimesh.Trimesh(gt_mesh.vertices, gt_mesh.faces)
    gt_mesh_rescale.vertices[:, 1] -= y_center
    gt_mesh_rescale.vertices = gt_mesh_rescale.vertices/(diag_max/2)
    latent_code_recover = fitting(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_codes, uv_vertices, uv_faces, gt_mesh_rescale)

    mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code_recover, uv_vertices, uv_faces, edges, which=which, resolution=x_res)
    
    mesh_sewing.vertices = mesh_sewing.vertices*diag_max/2 
    mesh_sewing.vertices[:, 1] += y_center
    mesh_atlas_f.vertices = mesh_atlas_f.vertices*diag_max/2 
    mesh_atlas_f.vertices[:, 1] += y_center
    mesh_atlas_b.vertices = mesh_atlas_b.vertices*diag_max/2 
    mesh_atlas_b.vertices[:, 1] += y_center


    sewing_path = os.path.join(save_path, '%s-sewing.obj'%save_name)
    atlas_path_f = os.path.join(save_path, '%s-atlas-f.obj'%save_name)
    atlas_path_b = os.path.join(save_path, '%s-atlas-b.obj'%save_name)
    mesh_sewing.export(sewing_path)
    mesh_atlas_f.export(atlas_path_f)
    mesh_atlas_b.export(atlas_path_b)

        

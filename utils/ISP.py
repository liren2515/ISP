import torch
import trimesh
import numpy as np

from utils import mesh_reader
from utils.sewing import sewing, sewing_pants, sewing_skirt

def reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code, uv_vertices, uv_faces, edges, resolution=256, which=''):
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
        vertices_new, faces_list = mesh_reader.read_mesh_from_sdf(uv_vertices_batch, uv_faces_torch_f, sdf_pred, edges, thresh=-5e-2)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]

        v_f = np.zeros((len(vertices_new_f), 3))
        v_b = np.zeros((len(vertices_new_b), 3))
        v_f[:, :2] = vertices_new_f.cpu().numpy()
        v_b[:, :2] = vertices_new_b.cpu().numpy()
        mesh_pattern_f = trimesh.Trimesh(v_f, faces_new_f.cpu().numpy())
        mesh_pattern_b = trimesh.Trimesh(v_b, faces_new_b.cpu().numpy())

        idx_boundary_v_f = mesh_reader.select_boundary(mesh_pattern_f)
        idx_boundary_v_b = mesh_reader.select_boundary(mesh_pattern_b)

        boundary_v_f = torch.FloatTensor(mesh_pattern_f.vertices[idx_boundary_v_f]).cuda()[:,:2]
        boundary_v_b = torch.FloatTensor(mesh_pattern_b.vertices[idx_boundary_v_b]).cuda()[:,:2]
        pred_f = model_sdf_f(boundary_v_f*10, latent_code[:len(boundary_v_f)])
        pred_b = model_sdf_b(boundary_v_b*10, latent_code[:len(boundary_v_b)])
        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1).cpu().numpy()
        label_b = torch.argmax(label_b, dim=-1).cpu().numpy()

        if which == 'tee':
            faces_sewing = sewing(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b)
        elif which == 'pants':
            faces_sewing = sewing_pants(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b)
        elif which == 'skirt':
            faces_sewing = sewing_skirt(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b)
        
        pattern_vertices_f = torch.FloatTensor(mesh_pattern_f.vertices).cuda()[:,:2]
        pattern_vertices_b = torch.FloatTensor(mesh_pattern_b.vertices).cuda()[:,:2]
        pred_atlas_f = model_atlas_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])/10

        mesh_atlas_f = trimesh.Trimesh(pred_atlas_f.cpu().numpy(), mesh_pattern_f.faces, process=False, valid=False)
        mesh_atlas_b = trimesh.Trimesh(pred_atlas_b.cpu().numpy(), mesh_pattern_b.faces, process=False, valid=False)

        num_vertices_f = len(mesh_atlas_f.vertices)
        vertices_sewing = np.concatenate((mesh_atlas_f.vertices, mesh_atlas_b.vertices), axis=0)
        faces_sewing = np.concatenate((mesh_atlas_f.faces, num_vertices_f+mesh_atlas_b.faces, faces_sewing), axis=0)
        mesh_sewing = trimesh.Trimesh(vertices_sewing, faces_sewing)

    return mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b

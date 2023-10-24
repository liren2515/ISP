import torch
import torch.nn.functional as F
import numpy as np
import trimesh


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
import torch
import torch.nn.functional as F
from utils.draping import uv_to_3D


def closest_points_with_normal(points1, points2, normals):
    vec = points1[:, :, None] - points2[:, None]
    dist = torch.sum(vec**2, dim=-1)
    dist_min, idx_closest_point = torch.min(dist, dim=-1)

    idx_closest_point = idx_closest_point.unsqueeze(-1).repeat(1,1,3)
    closest_point = torch.gather(points2, 1, idx_closest_point)
    closest_normals = torch.gather(normals, 1, idx_closest_point)
    return closest_point, closest_normals, dist_min, idx_closest_point

def uv_to_3D_inverse(vertices, faces_batch, barycentric_sewing_batch, closest_face_idx_sewing_batch):
    _B = len(closest_face_idx_sewing_batch)
    garment_faces_id = torch.gather(faces_batch, 1, closest_face_idx_sewing_batch.unsqueeze(-1).repeat(1,1,3))
    garment_faces_id = garment_faces_id.reshape(_B, -1)
    garment_faces_id = garment_faces_id.unsqueeze(-1).repeat(1,1,3).detach().cuda()

    vertices_triangles = torch.gather(vertices, 1, garment_faces_id)
    vertices_triangles = vertices_triangles.reshape(_B, -1, 3, 3)
    vertices_bary = (vertices_triangles * barycentric_sewing_batch[:, :, :, None]).sum(dim=-2)

    return vertices_bary

def generate_uv(points, indicator_unique_uv, res=200, bk=-1):
    _B = len(points)
    uv = torch.zeros((_B, res,res,3)).reshape(_B, -1,3).cuda() + bk
    uv[indicator_unique_uv] = points
    uv = uv.reshape(_B, res,res,3).permute(0, 3,1,2)
    return uv

def assamble_closest_points(garment_skinning_init, verts_body_closest, normals_body_closest, dist_body_min, verts_cloth_closest, normals_cloth_closest, dist_cloth_min, sigma=0.04, eps=5e-3, thresh_reg=1e-3, is_bottom=False):
    dis_bg = garment_skinning_init - verts_body_closest
    dis_gg = garment_skinning_init - verts_cloth_closest
    q_bg = (dis_bg*normals_body_closest).sum(dim=-1)
    q_gg = (dis_gg*normals_cloth_closest).sum(dim=-1)
    q_bg = F.relu(eps - q_bg)
    q_gg = F.relu(eps - q_gg)

    a_bg = torch.exp(-dist_body_min**2/(2*sigma**2))
    if is_bottom:
        a_gg = torch.exp(-dist_cloth_min**2/(2*0.004**2))
    else:
        a_gg = torch.exp(-dist_cloth_min**2/(2*sigma**2))

    f_bg = a_bg*q_bg
    f_gg = a_gg*q_gg

    indicator = f_gg < f_bg

    f_max = torch.maximum(f_gg, f_bg)

    f_bg = f_bg.unsqueeze(-1)*normals_body_closest
    f_gg = f_gg.unsqueeze(-1)*normals_cloth_closest

    f_gg[indicator] = f_bg[indicator]

    return f_gg

def compute_force(garment_skinning_init, verts_cloth_closest, normals_cloth_closest, dist_cloth_min, sigma=0.04, eps=5e-3, thresh_reg=1e-3):
    dis_gg = garment_skinning_init - verts_cloth_closest
    q_gg = (dis_gg*normals_cloth_closest).sum(dim=-1)
    q_gg = F.relu(eps - q_gg)

    a_gg = torch.exp(-dist_cloth_min**2/(2*sigma**2))
    f_gg = a_gg*q_gg
    f_gg = f_gg.unsqueeze(-1)*normals_cloth_closest

    return f_gg



def draping_layer(packed_input_u, packed_input_o, packed_input_atlas_o, packed_skinning_u, packed_skinning_o, pose, beta, model_diffusion, model_draping, model_layer, uv_faces, smpl_server, deformed_cloth, body=None, is_bottom=False):
    # vertices_garment_T - (#P, 3) 
    with torch.no_grad():

        garment_skinning_f_u, garment_skinning_b_u, garment_skinning_u, garment_faces_u = packed_skinning_u
        deformed_cloth.update_single(garment_skinning_u, garment_faces_u)
        
        garment_skinning_f_o, garment_skinning_b_o, garment_skinning_o, garment_faces_o = packed_skinning_o
        num_v_f_o = garment_skinning_f_o.shape[-2]

        barycentric_atlas_f, barycentric_atlas_b, closest_face_idx_atlas_f, closest_face_idx_atlas_b, input_pattern_f, input_pattern_b, indicator_unique_v_f, indicator_unique_v_b = packed_input_atlas_o
        pattern_f = input_pattern_f.unsqueeze(0)
        pattern_b = input_pattern_b.unsqueeze(0)
        barycentric_atlas_f = barycentric_atlas_f.unsqueeze(0)
        barycentric_atlas_b = barycentric_atlas_b.unsqueeze(0)
        closest_face_idx_atlas_f = closest_face_idx_atlas_f.unsqueeze(0)
        closest_face_idx_atlas_b = closest_face_idx_atlas_b.unsqueeze(0)
        indicator_unique_v_f = indicator_unique_v_f.unsqueeze(0)
        indicator_unique_v_b = indicator_unique_v_b.unsqueeze(0)
        faces_garment_f_o = torch.LongTensor(packed_input_o[2]).cuda().unsqueeze(0)
        faces_garment_b_o = torch.LongTensor(packed_input_o[3]).cuda().unsqueeze(0)

        verts_cloth_closest, normals_cloth_closest, dist_cloth_min, _ = closest_points_with_normal(garment_skinning_o, deformed_cloth.v, deformed_cloth.n)
        if body is not None:
            verts_body_closest, normals_body_closest, dist_body_min, _ = closest_points_with_normal(garment_skinning_o, body.vb, body.nb)
            force_closest = assamble_closest_points(garment_skinning_o, verts_body_closest, normals_body_closest, dist_body_min, verts_cloth_closest, normals_cloth_closest, dist_cloth_min, sigma=0.04, is_bottom=is_bottom)
        else:
            force_closest = compute_force(garment_skinning_o, verts_cloth_closest, normals_cloth_closest, dist_cloth_min, sigma=0.04)

        force_closest_f = force_closest[:, :num_v_f_o]
        force_closest_b = force_closest[:, num_v_f_o:]

        garment_skinning_init_uv_f = uv_to_3D_inverse(garment_skinning_f_o, faces_garment_f_o, barycentric_atlas_f, closest_face_idx_atlas_f)
        garment_skinning_init_uv_b = uv_to_3D_inverse(garment_skinning_b_o, faces_garment_b_o, barycentric_atlas_b, closest_face_idx_atlas_b)
        normals_closest_uv_f = uv_to_3D_inverse(force_closest_f, faces_garment_f_o, barycentric_atlas_f, closest_face_idx_atlas_f)
        normals_closest_uv_b = uv_to_3D_inverse(force_closest_b, faces_garment_b_o, barycentric_atlas_b, closest_face_idx_atlas_b)

        garment_skinning_init_uv_f = generate_uv(garment_skinning_init_uv_f, indicator_unique_v_f, res=200)
        garment_skinning_init_uv_b = generate_uv(garment_skinning_init_uv_b, indicator_unique_v_b, res=200)
        normals_closest_uv_f = generate_uv(normals_closest_uv_f, indicator_unique_v_f, res=200, bk=0)
        normals_closest_uv_b = generate_uv(normals_closest_uv_b, indicator_unique_v_b, res=200, bk=0)

        pattern_f = torch.cat((pattern_f, garment_skinning_init_uv_f, normals_closest_uv_f), dim=1)
        pattern_b = torch.cat((pattern_b, garment_skinning_init_uv_b, normals_closest_uv_b), dim=1)

        pattern_deform_f, pattern_deform_b = model_layer(pattern_f, pattern_b)
        
        fix_mask = packed_input_o[-2].unsqueeze(0)
        barycentric_uv_f, barycentric_uv_b, closest_face_idx_uv_f, closest_face_idx_uv_b, fix_mask = packed_input_o[4:9]
        barycentric_uv_batch_f = barycentric_uv_f.unsqueeze(0)
        barycentric_uv_batch_b = barycentric_uv_b.unsqueeze(0)
        closest_face_idx_uv_batch_f = closest_face_idx_uv_f.unsqueeze(0)
        closest_face_idx_uv_batch_b = closest_face_idx_uv_b.unsqueeze(0)
        pattern_deform_f = pattern_deform_f/100
        pattern_deform_b = pattern_deform_b/100
        pattern_deform_f = pattern_deform_f * fix_mask[None, None]
        pattern_deform_b = pattern_deform_b * fix_mask[None, None]

        pattern_deform_f = pattern_deform_f.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_b = pattern_deform_b.reshape(1, 3, -1).permute(0,2,1)
        pattern_deform_bary_f = uv_to_3D(pattern_deform_f, barycentric_uv_batch_f, closest_face_idx_uv_batch_f, uv_faces)
        pattern_deform_bary_b = uv_to_3D(pattern_deform_b, barycentric_uv_batch_b, closest_face_idx_uv_batch_b, uv_faces)

        garment_layer_f_o = garment_skinning_f_o + pattern_deform_bary_f
        garment_layer_b_o = garment_skinning_b_o + pattern_deform_bary_b

    return garment_layer_f_o.detach(), garment_layer_b_o.detach()

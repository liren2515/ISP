import os,sys
import numpy as np 
import torch
import trimesh
import argparse

from networks import SDF
from utils.ISP import reconstruct_batch
from utils import mesh_reader


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


    return model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, diag_max, y_center


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--which', default="tee", type=str, help="tee/pants/skirt")
    parser.add_argument('--res', default=256, type=int, help="resolution")
    parser.add_argument('--save_path', type=str, default='./tmp')
    parser.add_argument('--save_name', type=str, default='tee')
    parser.add_argument('--idx_G', default=0, type=int, help="index of garment")

    args = parser.parse_args()

    which = args.which
    save_path = args.save_path
    save_name = args.save_name
    x_res = y_res = args.res
    idx_G = args.idx_G

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_sdf_f, model_sdf_b, model_rep, model_atlas_f, model_atlas_b, diag_max, y_center = load_model(which)

    uv_vertices, uv_faces = mesh_reader.create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
    edges = torch.LongTensor(mesh_uv.edges).cuda()
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()

    latent_codes = model_rep.weights.detach()

    # change idx_G to infer different garments
    latent_code = latent_codes[idx_G]

    mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = reconstruct_batch(model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b, latent_code, uv_vertices, uv_faces, edges, which=which, resolution=x_res)
    
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
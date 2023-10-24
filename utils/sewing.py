import numpy as np
import trimesh
from utils import mesh_reader

def sewing(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b):
    faces_sewing = []
    # sewing left body
    num_vertices_f = len(mesh_pattern_f.vertices)
    idx_boundary_body_left_f, idx_boundary_body_right_f, idx_highest_left_f, idx_highest_right_f = mesh_reader.get_body_idx(mesh_pattern_f, idx_boundary_v_f, label_f, front=True)
    idx_boundary_body_left_b, idx_boundary_body_right_b, idx_highest_left_b, idx_highest_right_b = mesh_reader.get_body_idx(mesh_pattern_b, idx_boundary_v_b, label_b, front=False)

    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_left_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_left_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_body_left = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_left_f, idx_boundary_body_left_b, num_vertices_f, xy='y')

    # sewing right body
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_right_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_right_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_body_right = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_right_f, idx_boundary_body_right_b, num_vertices_f, xy='y', reverse=True)
    faces_sewing = [boundary_faces_body_left, boundary_faces_body_right]

    # sewing bottom left arm
    idx_boundary_up_left_arm_f, idx_boundary_bottom_left_arm_f, idx_boundary_up_right_arm_f, idx_boundary_bottom_right_arm_f, idx_closest_left_bottom_f, idx_closest_right_bottom_f = mesh_reader.get_sleeve_idx(mesh_pattern_f, idx_boundary_v_f, label_f, front=True)
    idx_boundary_up_left_arm_b, idx_boundary_bottom_left_arm_b, idx_boundary_up_right_arm_b, idx_boundary_bottom_right_arm_b, idx_closest_left_bottom_b, idx_closest_right_bottom_b = mesh_reader.get_sleeve_idx(mesh_pattern_b, idx_boundary_v_b, label_b, front=False)

    boundary_bottom_left_arm_f = mesh_pattern_f.vertices[idx_boundary_bottom_left_arm_f][:,:2]
    boundary_bottom_left_arm_b = mesh_pattern_b.vertices[idx_boundary_bottom_left_arm_b][:,:2]
    boundary_up_left_arm_f = mesh_pattern_f.vertices[idx_boundary_up_left_arm_f][:,:2]
    boundary_up_left_arm_b = mesh_pattern_b.vertices[idx_boundary_up_left_arm_b][:,:2]
    boundary_bottom_right_arm_f = mesh_pattern_f.vertices[idx_boundary_bottom_right_arm_f][:,:2]
    boundary_bottom_right_arm_b = mesh_pattern_b.vertices[idx_boundary_bottom_right_arm_b][:,:2]
    boundary_up_right_arm_f = mesh_pattern_f.vertices[idx_boundary_up_right_arm_f][:,:2]
    boundary_up_right_arm_b = mesh_pattern_b.vertices[idx_boundary_up_right_arm_b][:,:2]

    
    boundary_up_left_arm_b[:, 1] += 0.5
    boundary_faces_up_left_arm = mesh_reader.triangulation_2D(boundary_up_left_arm_f, boundary_up_left_arm_b, idx_boundary_up_left_arm_f, idx_boundary_up_left_arm_b, num_vertices_f, xy='x', reverse=True)

    boundary_up_right_arm_b[:, 1] += 0.5
    boundary_faces_up_right_arm = mesh_reader.triangulation_2D(boundary_up_right_arm_f, boundary_up_right_arm_b, idx_boundary_up_right_arm_f, idx_boundary_up_right_arm_b, num_vertices_f, xy='x', reverse=True)

    faces_sewing += [boundary_faces_up_left_arm, boundary_faces_up_right_arm]
    
    if idx_boundary_bottom_left_arm_f is not None:
        boundary_bottom_left_arm_b[:, 1] += 0.5
        boundary_faces_bottom_left_arm = mesh_reader.triangulation_2D(boundary_bottom_left_arm_f, boundary_bottom_left_arm_b, idx_boundary_bottom_left_arm_f, idx_boundary_bottom_left_arm_b, num_vertices_f, xy='x')
        boundary_bottom_right_arm_b[:, 1] += 0.5
        boundary_faces_bottom_right_arm = mesh_reader.triangulation_2D(boundary_bottom_right_arm_f, boundary_bottom_right_arm_b, idx_boundary_bottom_right_arm_f, idx_boundary_bottom_right_arm_b, num_vertices_f, xy='x')
        

        faces_sewing += [boundary_faces_bottom_left_arm, boundary_faces_bottom_right_arm]

        faces_extra = np.array([[idx_closest_left_bottom_f, idx_highest_left_f, idx_highest_left_b+num_vertices_f],
                                [idx_highest_left_b+num_vertices_f, idx_closest_left_bottom_b+num_vertices_f, idx_closest_left_bottom_f],
                                [idx_closest_right_bottom_b+num_vertices_f, idx_highest_right_b+num_vertices_f, idx_highest_right_f],
                                [idx_highest_right_f, idx_closest_right_bottom_f, idx_closest_right_bottom_b+num_vertices_f]])

        faces_sewing.append(faces_extra)
    
    faces_sewing = np.concatenate(faces_sewing, axis=0)

    return faces_sewing

def sewing_pants(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b):
    faces_sewing = []

    num_vertices_f = len(mesh_pattern_f.vertices)
    idx_boundary_body_left_left_f, idx_boundary_body_left_right_f, idx_boundary_body_right_right_f, idx_boundary_body_right_left_f, idx_highest_left_right_f, idx_highest_right_left_f = mesh_reader.get_pants_idx(mesh_pattern_f, idx_boundary_v_f, label_f)
    idx_boundary_body_left_left_b, idx_boundary_body_left_right_b, idx_boundary_body_right_right_b, idx_boundary_body_right_left_b, idx_highest_left_right_b, idx_highest_right_left_b = mesh_reader.get_pants_idx(mesh_pattern_b, idx_boundary_v_b, label_b)

    # sewing left left leg
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_left_left_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_left_left_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_leg_left_left = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_left_left_f, idx_boundary_body_left_left_b, num_vertices_f, xy='y')

    # sewing left right leg
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_left_right_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_left_right_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_leg_left_right = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_left_right_f, idx_boundary_body_left_right_b, num_vertices_f, xy='y')[:,[0,2,1]]

    # sewing right right leg
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_right_right_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_right_right_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_leg_right_right = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_right_right_f, idx_boundary_body_right_right_b, num_vertices_f, xy='y', reverse=True)

    # sewing right left leg
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_right_left_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_right_left_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_leg_right_left = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_right_left_f, idx_boundary_body_right_left_b, num_vertices_f, xy='y', reverse=True)[:,[0,2,1]]

    faces_sewing = [boundary_faces_leg_left_left, boundary_faces_leg_left_right, boundary_faces_leg_right_right, boundary_faces_leg_right_left]

    faces_extra = np.array([[idx_highest_right_left_f, idx_highest_left_right_f, idx_highest_right_left_b+num_vertices_f],
                            [idx_highest_left_right_b+num_vertices_f, idx_highest_right_left_b+num_vertices_f, idx_highest_left_right_f]])
    faces_extra = faces_extra[:,[0,2,1]]

    
    faces_sewing.append(faces_extra)
    
    faces_sewing = np.concatenate(faces_sewing, axis=0)

    return faces_sewing

def sewing_skirt(mesh_pattern_f, mesh_pattern_b, idx_boundary_v_f, idx_boundary_v_b, label_f, label_b):
    faces_sewing = []
    # sewing left body
    num_vertices_f = len(mesh_pattern_f.vertices)
    idx_boundary_body_left_f, idx_boundary_body_right_f = mesh_reader.get_skirt_idx(mesh_pattern_f, idx_boundary_v_f, label_f)
    idx_boundary_body_left_b, idx_boundary_body_right_b = mesh_reader.get_skirt_idx(mesh_pattern_b, idx_boundary_v_b, label_b)

    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_left_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_left_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_body_left = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_left_f, idx_boundary_body_left_b, num_vertices_f, xy='y')

    # sewing right body
    boundary_f = mesh_pattern_f.vertices[idx_boundary_body_right_f][:,:2]
    boundary_b = mesh_pattern_b.vertices[idx_boundary_body_right_b][:,:2]
    boundary_b[:, 0] += 0.5
    boundary_faces_body_right = mesh_reader.triangulation_2D(boundary_f, boundary_b, idx_boundary_body_right_f, idx_boundary_body_right_b, num_vertices_f, xy='y', reverse=True)
    faces_sewing = [boundary_faces_body_left, boundary_faces_body_right]
    
    faces_sewing = np.concatenate(faces_sewing, axis=0)

    return faces_sewing

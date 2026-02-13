# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp

from . import lighting, ray_cast, textures
from .types import RenderOrder, RenderShapeType


@wp.func
def tid_to_coord_tiled(
    tid: wp.int32,
    num_cameras: wp.int32,
    width: wp.int32,
    height: wp.int32,
    tile_width: wp.int32,
    tile_height: wp.int32,
):
    num_pixels_per_view = width * height
    num_pixels_per_tile = tile_width * tile_height
    num_tiles_per_row = width // tile_width

    pixel_idx = tid % num_pixels_per_view
    view_idx = tid // num_pixels_per_view

    world_index = view_idx // num_cameras
    camera_index = view_idx % num_cameras

    tile_idx = pixel_idx // num_pixels_per_tile
    tile_pixel_idx = pixel_idx % num_pixels_per_tile

    tile_y = tile_idx // num_tiles_per_row
    tile_x = tile_idx % num_tiles_per_row

    py = tile_y * tile_height + tile_pixel_idx // tile_width
    px = tile_x * tile_width + tile_pixel_idx % tile_width

    return world_index, camera_index, py, px


@wp.func
def tid_to_coord_pixel_priority(tid: wp.int32, num_worlds: wp.int32, num_cameras: wp.int32, width: wp.int32):
    num_views_per_pixel = num_worlds * num_cameras

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_index = view_idx % num_worlds
    camera_index = view_idx // num_worlds

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, camera_index, py, px


@wp.func
def tid_to_coord_view_priority(tid: wp.int32, num_cameras: wp.int32, width: wp.int32, height: wp.int32):
    num_pixels_per_view = width * height

    pixel_idx = tid % num_pixels_per_view
    view_idx = tid // num_pixels_per_view

    world_index = view_idx // num_cameras
    camera_index = view_idx % num_cameras

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, camera_index, py, px


@wp.func
def pack_rgba_to_uint32(rgb: wp.vec3f, alpha: wp.float32) -> wp.uint32:
    """Pack RGBA values into a single uint32 for efficient memory access."""
    return (
        (wp.uint32(alpha * 255.0) << wp.uint32(24))
        | (wp.uint32(rgb[2] * 255.0) << wp.uint32(16))
        | (wp.uint32(rgb[1] * 255.0) << wp.uint32(8))
        | wp.uint32(rgb[0] * 255.0)
    )


@wp.kernel(enable_backward=False)
def render_megakernel(
    # Model and Options
    num_worlds: wp.int32,
    num_cameras: wp.int32,
    num_lights: wp.int32,
    img_width: wp.int32,
    img_height: wp.int32,
    render_order: wp.int32,
    tile_width: wp.int32,
    tile_height: wp.int32,
    enable_shadows: wp.bool,
    enable_textures: wp.bool,
    enable_ambient_lighting: wp.bool,
    enable_particles: wp.bool,
    enable_backface_culling: wp.bool,
    enable_global_world: wp.bool,
    max_distance: wp.float32,
    # Camera
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
    # Shapes BVH
    bvh_shapes_size: wp.int32,
    bvh_shapes_id: wp.uint64,
    bvh_shapes_group_roots: wp.array(dtype=wp.int32),
    # Shapes
    shape_enabled: wp.array(dtype=wp.uint32),
    shape_types: wp.array(dtype=wp.int32),
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_materials: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_colors: wp.array(dtype=wp.vec4f),
    shape_roughness: wp.array(dtype=wp.float32),
    shape_metallic: wp.array(dtype=wp.float32),
    shape_transforms: wp.array(dtype=wp.transformf),
    # Meshes
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_face_offsets: wp.array(dtype=wp.int32),
    mesh_face_vertices: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    # Mesh vertex normals
    mesh_vertex_normals: wp.array(dtype=wp.vec3f),
    mesh_vertex_normal_offsets: wp.array(dtype=wp.int32),
    # Particle BVH
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    # Particles
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    # Triangle Mesh:
    triangle_mesh_id: wp.uint64,
    triangle_mesh_color: wp.vec4f,
    # Particles color:
    particles_color: wp.vec4f,
    # Materials
    material_texture_ids: wp.array(dtype=wp.int32),
    material_texture_repeat: wp.array(dtype=wp.vec2f),
    material_rgba: wp.array(dtype=wp.vec4f),
    # Textures
    texture_offsets: wp.array(dtype=wp.int32),
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.array(dtype=wp.int32),
    texture_width: wp.array(dtype=wp.int32),
    # Lights
    light_active: wp.array(dtype=wp.bool),
    light_type: wp.array(dtype=wp.int32),
    light_cast_shadow: wp.array(dtype=wp.bool),
    light_positions: wp.array(dtype=wp.vec3f),
    light_orientations: wp.array(dtype=wp.vec3f),
    # Enabled Output
    render_color: wp.bool,
    render_depth: wp.bool,
    render_shape_index: wp.bool,
    render_normal: wp.bool,
    render_albedo: wp.bool,
    # Outputs
    out_pixels: wp.array(dtype=wp.uint32),
    out_depth: wp.array(dtype=wp.float32),
    out_shape_index: wp.array(dtype=wp.uint32),
    out_normal: wp.array(dtype=wp.vec3f),
    out_albedo: wp.array(dtype=wp.uint32),
):
    tid = wp.tid()

    if render_order == RenderOrder.PIXEL_PRIORITY:
        world_index, camera_index, py, px = tid_to_coord_pixel_priority(tid, num_worlds, num_cameras, img_width)
    elif render_order == RenderOrder.VIEW_PRIORITY:
        world_index, camera_index, py, px = tid_to_coord_view_priority(tid, num_cameras, img_width, img_height)
    elif render_order == RenderOrder.TILED:
        world_index, camera_index, py, px = tid_to_coord_tiled(
            tid, num_cameras, img_width, img_height, tile_width, tile_height
        )
    else:
        return

    if px >= img_width or py >= img_height:
        return

    pixels_per_camera = img_width * img_height
    pixels_per_world = num_cameras * pixels_per_camera
    out_index = world_index * pixels_per_world + camera_index * pixels_per_camera + py * img_width + px

    ray_origin_world = wp.transform_point(
        camera_transforms[camera_index, world_index], camera_rays[camera_index, py, px, 0]
    )
    ray_dir_world = wp.transform_vector(
        camera_transforms[camera_index, world_index], camera_rays[camera_index, py, px, 1]
    )

    closest_hit = ray_cast.closest_hit(
        bvh_shapes_size,
        bvh_shapes_id,
        bvh_shapes_group_roots,
        bvh_particles_size,
        bvh_particles_id,
        bvh_particles_group_roots,
        world_index,
        enable_global_world,
        enable_particles,
        enable_backface_culling,
        max_distance,
        shape_enabled,
        shape_types,
        shape_mesh_indices,
        shape_sizes,
        shape_transforms,
        mesh_ids,
        particles_position,
        particles_radius,
        triangle_mesh_id,
        ray_origin_world,
        ray_dir_world,
    )

    if closest_hit.shape_index == ray_cast.NO_HIT_SHAPE_ID:
        return

    # Smooth vertex normal interpolation for mesh shapes
    shading_normal = closest_hit.normal
    if closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        if shape_types[closest_hit.shape_index] == RenderShapeType.MESH:
            mesh_id = closest_hit.shape_mesh_index
            f = closest_hit.face_idx
            if mesh_id >= 0 and f >= 0 and mesh_vertex_normal_offsets.shape[0] > 0 and mesh_vertex_normals.shape[0] > 0:
                normal_base = mesh_vertex_normal_offsets[mesh_id]
                if mesh_vertex_normals.shape[0] > normal_base:
                    # Winding convention matches textures: (f*3+2, f*3+0, f*3+1)
                    v_idx = wp.vec3i(f * 3 + 2, f * 3 + 0, f * 3 + 1)
                    n0 = mesh_vertex_normals[normal_base + v_idx[0]]
                    n1 = mesh_vertex_normals[normal_base + v_idx[1]]
                    n2 = mesh_vertex_normals[normal_base + v_idx[2]]
                    bw = 1.0 - closest_hit.bary_u - closest_hit.bary_v
                    smooth_n = n0 * bw + n1 * closest_hit.bary_u + n2 * closest_hit.bary_v
                    len_sn = wp.length(smooth_n)
                    if len_sn > 0.0:
                        shading_normal = wp.normalize(smooth_n)

    if render_depth:
        out_depth[out_index] = closest_hit.distance

    if render_normal:
        out_normal[out_index] = shading_normal

    if render_shape_index:
        out_shape_index[out_index] = closest_hit.shape_index

    if not render_color and not render_albedo:
        return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * closest_hit.distance

    color = wp.vec4f(1.0)
    if closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        color = shape_colors[closest_hit.shape_index]
        if shape_materials[closest_hit.shape_index] > -1:
            color = wp.cw_mul(color, material_rgba[shape_materials[closest_hit.shape_index]])
    elif closest_hit.shape_index == ray_cast.TRIANGLE_MESH_SHAPE_ID:
        color = triangle_mesh_color
    elif closest_hit.shape_index == ray_cast.PARTICLES_SHAPE_ID:
        color = particles_color

    base_color = wp.vec3f(color[0], color[1], color[2])
    out_color = wp.vec3f(0.0)

    if enable_textures and closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        material_index = shape_materials[closest_hit.shape_index]
        if material_index > -1:
            texture_index = material_texture_ids[material_index]
            if texture_index > -1:
                tex_color = textures.sample_texture(
                    shape_types[closest_hit.shape_index],
                    shape_transforms[closest_hit.shape_index],
                    material_index,
                    texture_index,
                    material_texture_repeat[material_index],
                    texture_offsets[texture_index],
                    texture_data,
                    texture_height[texture_index],
                    texture_width[texture_index],
                    mesh_face_offsets,
                    mesh_face_vertices,
                    mesh_texcoord,
                    mesh_texcoord_offsets,
                    hit_point,
                    closest_hit.bary_u,
                    closest_hit.bary_v,
                    closest_hit.face_idx,
                    closest_hit.shape_mesh_index,
                )

                base_color = wp.vec3f(
                    base_color[0] * tex_color[0],
                    base_color[1] * tex_color[1],
                    base_color[2] * tex_color[2],
                )

    if render_albedo:
        out_albedo[out_index] = pack_rgba_to_uint32(base_color, 1.0)

    if not render_color:
        return

    # Derive PBR material parameters from per-shape roughness/metallic
    roughness = wp.float32(0.5)
    metallic = wp.float32(0.0)
    if closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        roughness = wp.clamp(shape_roughness[closest_hit.shape_index], 0.0, 1.0)
        metallic = wp.clamp(shape_metallic[closest_hit.shape_index], 0.0, 1.0)

    # Shininess from roughness (matching GL shader)
    gloss = 1.0 - roughness
    shininess = 1.0 + wp.pow(gloss, 4.0) * 1023.0

    # F0: dielectrics=0.04, metals=albedo (vec3 for metal tinting)
    f0 = wp.vec3f(
        0.04 * (1.0 - metallic) + base_color[0] * metallic,
        0.04 * (1.0 - metallic) + base_color[1] * metallic,
        0.04 * (1.0 - metallic) + base_color[2] * metallic,
    )

    if enable_ambient_lighting:
        up = wp.vec3f(0.0, 0.0, 1.0)
        len_n = wp.length(shading_normal)
        n = shading_normal if len_n > 0.0 else up
        n = wp.normalize(n)
        hemispheric = 0.5 * (wp.dot(n, up) + 1.0)
        sky = wp.vec3f(0.4, 0.4, 0.45)
        ground = wp.vec3f(0.1, 0.1, 0.12)
        ambient_color = sky * hemispheric + ground * (1.0 - hemispheric)
        ambient_intensity = 0.5
        metal_ambient_boost = 1.0 + 0.25 * metallic * (1.0 - 0.5 * roughness)
        out_color = wp.vec3f(
            base_color[0] * (ambient_color[0] * ambient_intensity) * metal_ambient_boost,
            base_color[1] * (ambient_color[1] * ambient_intensity) * metal_ambient_boost,
            base_color[2] * (ambient_color[2] * ambient_intensity) * metal_ambient_boost,
        )

    # Apply lighting and shadows
    view_dir = -ray_dir_world
    for light_index in range(num_lights):
        light_contrib = lighting.compute_lighting(
            enable_shadows,
            enable_particles,
            enable_backface_culling,
            world_index,
            enable_global_world,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            shape_enabled,
            shape_types,
            shape_mesh_indices,
            shape_sizes,
            shape_transforms,
            mesh_ids,
            light_active[light_index],
            light_type[light_index],
            light_cast_shadow[light_index],
            light_positions[light_index],
            light_orientations[light_index],
            particles_position,
            particles_radius,
            triangle_mesh_id,
            shading_normal,
            hit_point,
            view_dir,
            shininess,
        )
        diffuse_contrib = light_contrib[0] * (1.0 - metallic)
        out_color = wp.vec3f(
            out_color[0] + base_color[0] * diffuse_contrib + f0[0] * light_contrib[1],
            out_color[1] + base_color[1] * diffuse_contrib + f0[1] * light_contrib[1],
            out_color[2] + base_color[2] * diffuse_contrib + f0[2] * light_contrib[1],
        )

    out_color = wp.min(wp.max(out_color, wp.vec3f(0.0)), wp.vec3f(1.0))
    out_pixels[out_index] = pack_rgba_to_uint32(out_color, 1.0)

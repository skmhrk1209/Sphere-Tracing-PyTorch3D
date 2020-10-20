import torch
import torchvision
import numpy as np
from torch import nn
from torch import autograd
from pytorch3d import renderer
from modules import *
import logging
import sdf


def main():

    device = torch.device("cuda:0")
    image_size = (256, 256)

    R, T = renderer.look_at_view_transform(
        dist=(10.,),
        elev=(np.pi / 5.,),
        azim=(np.pi / 4.,),
        degrees=False,
        device=device,
    )
    cameras = renderer.PerspectiveCameras(
        R=R, T=T, 
        device=device,
    )

    signed_distance_functions = [
        lambda p: sdf.sphere(p, 5.),
        # lambda p: sdf.round(sdf.box(p, 4.), 0.2),
        lambda p: sdf.union(sdf.sphere(p, 5.), sdf.round(sdf.box(p, 4.), 0.2), 0.2),
        lambda p: sdf.intersection(sdf.sphere(p, 5.), sdf.round(sdf.box(p, 4.), 0.2), 0.2),
        lambda p: sdf.subtraction(sdf.sphere(p, 5.), sdf.round(sdf.box(p, 4.), 0.2), 0.2),
        lambda p: sdf.union(sdf.sphere(p - p.new_tensor([[[0., 2., 0.]]]), 3.), sdf.round(sdf.box(p, p.new_tensor([[[4., 1., 4.]]])), 0.2), 0.2),
        lambda p: sdf.subtraction(sdf.sphere(p - p.new_tensor([[[0., 2., 0.]]]), 3.), sdf.round(sdf.box(p, p.new_tensor([[[4., 1., 4.]]])), 0.2), 0.2),
        lambda p: sdf.torus(p, p.new_tensor([[[4., 2.]]])),
        lambda p: sdf.torus(sdf.twist(p, 0.4), p.new_tensor([[[4., 2.]]])),
        # lambda p: sdf.round(sdf.box(sdf.bend(p, 0.3), p.new_tensor([[[8., 2., 2.]]])), 0.2),
        # lambda p: sdf.sphere(sdf.repetition(p, 2), 0.5),
        # lambda p: sdf.round(sdf.box(sdf.repetition(p, 2), 0.4), 0.2),
    ]

    images = []

    for signed_distance_function in signed_distance_functions:

        y = torch.linspace(1., -1., image_size[-2], device=device)
        x = torch.linspace(1., -1., image_size[-1], device=device)
        y, x = torch.meshgrid(y, x)
        z = torch.ones_like(y)
        positions = torch.stack((x, y, z), dim=-1)
        positions = positions.reshape(1, -1, 3)
        positions = cameras.unproject_points(positions, world_coordinates=True)
        directions = positions - cameras.get_camera_center().reshape(-1, 1, 3)
        directions = nn.functional.normalize(directions, dim=-1)

        lights = renderer.DirectionalLights(
            ambient_color=torch.full((1, 3), 0.2) + (torch.rand(1, 3) * 2 - 1) * 0.04,
            diffuse_color=torch.full((1, 3), 0.8) + (torch.rand(1, 3) * 2 - 1) * 0.16,
            specular_color=((1., 1., 1.),),
            direction=((0.8, 0.2, 0.),),
            device=device,
        )
        materials = renderer.Materials(
            ambient_color=((1., 1., 1.),),
            diffuse_color=((1., 1., 1.),),
            specular_color=((1., 1., 1.),),
            device=device,
        )

        positions, converged = SphereTracing.apply(signed_distance_function, positions, directions, None, 1000, 1e-3)
        normals = compute_normal(signed_distance_function, positions, 0.0)
        textures = torch.ones_like(positions)

        renderings = phong_shading(positions, normals, textures, cameras, lights, materials)
        renderings = torch.where(converged, renderings, torch.zeros_like(renderings))
        renderings = torch.min(renderings, torch.ones_like(renderings))
        renderings = renderings.reshape(*image_size, 3)
        renderings = renderings.permute(2, 0, 1)
        
        images.append(renderings)

    images = torch.stack(images)
    torchvision.utils.save_image(images, "result.png", nrow=4)


if __name__ == "__main__":
    main()

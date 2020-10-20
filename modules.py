import torch
from torch import nn
from torch import autograd


class SphereTracing(autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        signed_distance_function, 
        positions, 
        directions, 
        foreground_masks, 
        num_iterations, 
        convergence_threshold,
        *parameters,
    ):
        # vanilla sphere tracing
        with torch.no_grad():
            positions, converged = sphere_tracing(
                signed_distance_function=signed_distance_function, 
                positions=positions, 
                directions=directions, 
                foreground_masks=foreground_masks,
                num_iterations=num_iterations, 
                convergence_threshold=convergence_threshold,
            )
            positions = torch.where(converged, positions, torch.zeros_like(positions))

        # save tensors for backward pass
        ctx.save_for_backward(positions, directions, foreground_masks, converged)
        ctx.signed_distance_function = signed_distance_function
        ctx.parameters = parameters

        return positions, converged

    @staticmethod
    def backward(ctx, grad_outputs, *_):
        
        # restore tensors from forward pass
        positions, directions, foreground_masks, converged = ctx.saved_tensors
        signed_distance_function = ctx.signed_distance_function
        parameters = ctx.parameters

        # compute gradients using implicit differentiation
        with torch.enable_grad():
            positions = positions.detach()
            positions.requires_grad_(True)
            signed_distances = signed_distance_function(positions)
            grad_positions, = autograd.grad(
                outputs=signed_distances, 
                inputs=positions, 
                grad_outputs=torch.ones_like(signed_distances), 
                retain_graph=True,
            )
            grad_outputs_dot_directions = torch.sum(grad_outputs * directions, dim=-1, keepdim=True)
            grad_positions_dot_directions = torch.sum(grad_positions * directions, dim=-1, keepdim=True)
            # NOTE: avoid division by zero
            grad_positions_dot_directions = torch.where(
                grad_positions_dot_directions > 0,
                torch.max(grad_positions_dot_directions, torch.full_like(grad_positions_dot_directions, +1e-6)),
                torch.min(grad_positions_dot_directions, torch.full_like(grad_positions_dot_directions, -1e-6)),
            )
            grad_outputs = -grad_outputs_dot_directions / grad_positions_dot_directions
            # NOTE: zero gradient for unconverged points 
            grad_outputs = torch.where(converged, grad_outputs, torch.zeros_like(grad_outputs))
            grad_parameters = autograd.grad(
                outputs=signed_distances, 
                inputs=parameters, 
                grad_outputs=grad_outputs, 
                retain_graph=True,
            )

        return (None, None, None, None, None, None, *grad_parameters)


def sphere_tracing(
    signed_distance_function, 
    positions, 
    directions, 
    foreground_masks, 
    num_iterations, 
    convergence_threshold,
):
    for i in range(num_iterations):
        signed_distances = signed_distance_function(positions)
        if i:
            positions = torch.where(converged, positions, positions + directions * signed_distances)
        else:
            positions = positions + directions * signed_distances
        converged = torch.abs(signed_distances) < convergence_threshold
        if torch.all(converged[foreground_masks] if foreground_masks else converged):
            break

    return positions, converged


def compute_normal(signed_distance_function, positions, finite_difference_epsilon):

    if finite_difference_epsilon:
        finite_difference_epsilon = positions.new_tensor(finite_difference_epsilon)
        finite_difference_epsilon = finite_difference_epsilon.reshape(1, 1, 1)
        finite_difference_epsilon_x = nn.functional.pad(finite_difference_epsilon, (0, 2))
        finite_difference_epsilon_y = nn.functional.pad(finite_difference_epsilon, (1, 1))
        finite_difference_epsilon_z = nn.functional.pad(finite_difference_epsilon, (2, 0))
        normals_x = signed_distance_function(positions + finite_difference_epsilon_x) - signed_distance_function(positions - finite_difference_epsilon_x)
        normals_y = signed_distance_function(positions + finite_difference_epsilon_y) - signed_distance_function(positions - finite_difference_epsilon_y)
        normals_z = signed_distance_function(positions + finite_difference_epsilon_z) - signed_distance_function(positions - finite_difference_epsilon_z)
        normals = torch.cat((normals_x, normals_y, normals_z), dim=-1)

    else:
        create_graph = positions.requires_grad
        positions.requires_grad_(True)
        with torch.enable_grad():
            signed_distances = signed_distance_function(positions)
            normals, = autograd.grad(
                outputs=signed_distances, 
                inputs=positions, 
                grad_outputs=torch.ones_like(signed_distances),
                create_graph=create_graph,
            )
            
    return normals


def phong_shading(positions, normals, textures, cameras, lights, materials):
    light_diffuse_color = lights.diffuse(
        normals=normals, 
        points=positions,
    )
    light_specular_color = lights.specular(
        normals=normals,
        points=positions,
        camera_position=cameras.get_camera_center(),
        shininess=materials.shininess,
    )
    ambient_colors = materials.ambient_color * lights.ambient_color
    diffuse_colors = materials.diffuse_color * light_diffuse_color
    specular_colors = materials.specular_color * light_specular_color
    # NOTE: pytorch3d.renderer.phong_shading should be fixed as well
    assert diffuse_colors.shape == specular_colors.shape
    ambient_colors = ambient_colors.reshape(-1, *[1] * len(diffuse_colors.shape[1:-1]), 3)
    colors = (ambient_colors + diffuse_colors) * textures + specular_colors
    return colors

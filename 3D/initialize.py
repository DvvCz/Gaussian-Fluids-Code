import torch
import torch.nn.functional as F
import time
import os

from init_cond import *


def fit_velocity_with_gradient(gaussian_velocity: GaussianSplatting3DFast, reference_field, reference_gradient, data_generator, batch_size=8192, max_epoch=3000, verbose=1):
	def vorticity(x):
		g = gaussian_velocity.gradient(x)
		vor = torch.zeros((x.shape[0], 3), device=device)
		vor[:, 0] = g[:, 2, 1] - g[:, 1, 2]
		vor[:, 1] = g[:, 0, 2] - g[:, 2, 0]
		vor[:, 2] = g[:, 1, 0] - g[:, 0, 1]
		return vor
	
	st_time = time.time()
	gaussian_velocity.initialize_optimizers()
	for epoch in range(max_epoch):
		# calculate difference between gaussian_splatting and reference_field
		data = data_generator(batch_size)
		ref_val = reference_field(data)		# reference_field may change data, must execute first
		ref_grad = reference_gradient(data)
		val, grad = gaussian_velocity.get_losses(data, ref_val=ref_val, weight_val=1., ref_grad=ref_grad, weight_grad=1.)
		scalings_ratio = torch.exp(-gaussian_velocity.scalings.min(axis=-1).values + gaussian_velocity.scalings.max(axis=-1).values)
		volumes = torch.exp(-gaussian_velocity.scalings.sum(axis=-1))
		loss = F.l1_loss(val, ref_val)
		loss_grad = F.l1_loss(grad, ref_grad)
		loss_aniso = (torch.where(scalings_ratio >= 1.5, scalings_ratio, 1.5) - 1.5).mean()
		loss_vol = ((volumes / volumes.mean() - 1) ** 2).mean()
		(loss_aniso + loss_vol).backward()
		gaussian_velocity.step(loss + loss_grad + loss_aniso + loss_vol)
		
		if verbose and epoch % 100 == 0:
			with torch.no_grad():
				en_time = time.time()
				print(f'loss: {loss}, loss_grad: {loss_grad}, loss_aniso: {loss_aniso}, loss_vol: {loss_vol}, divergence: {(grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2]).abs().mean()}')
				print('time:', en_time - st_time)
				st_time = time.time()
	
	if verbose:
		with torch.no_grad():
			en_time = time.time()
			print(f'loss: {loss}, loss_grad: {loss_grad}, loss_aniso: {loss_aniso}, loss_vol: {loss_vol}, divergence: {(grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2]).abs().mean()}')
			print('time:', en_time - st_time)


def SimulationInitialize():
	x_min, x_max, y_min, y_max, z_min, z_max = domain[cmd_args.init_cond]
	x_Nvis, y_Nvis, z_Nvis = visualize_res[cmd_args.init_cond]
	
	velocity_ref = eval(cmd_args.init_cond)
	def vorticity_ref(x):
		g = velocity_ref.gradient(x)
		vor = torch.zeros((x.shape[0], 3), device=device)
		vor[:, 0] = g[:, 2, 1] - g[:, 1, 2]
		vor[:, 1] = g[:, 0, 2] - g[:, 2, 0]
		vor[:, 2] = g[:, 1, 0] - g[:, 0, 1]
		return vor
	def divergence_ref(x):
		return torch.diagonal(velocity_ref.gradient(x), dim1=-2, dim2=-1).sum(dim=-1)
	
	write_vti(lambda x: (velocity_ref(x) ** 2).sum(axis=-1) ** .5, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'velocity_ref.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	write_vti(lambda x: (vorticity_ref(x) ** 2).sum(axis=-1) ** .5, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'vorticity_ref.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	write_vti(divergence_ref, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'divergence_ref.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	write_vti(lambda x: (vorticity_ref(x) * velocity_ref(x)).sum(axis=-1), x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'helicity_ref.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	
	x_Np, y_Np, z_Np = initial_particle_count[cmd_args.init_cond]
	gaussian_velocity = GaussianSplatting3DFast(x_min, x_max, y_min, y_max, z_min, z_max, get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_Np, y_Np, z_Np).cpu().numpy(), dim=3)
	print('Particle count:', gaussian_velocity.N)
	
	def default_generator(n):
		return torch.rand_like(gaussian_velocity.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=device) + torch.tensor([x_min, y_min, z_min], device=device)
	def generate_gaussians(n):
		pick = torch.randint(0, gaussian_velocity.N, (n,), device=device)
		with torch.no_grad():
			variances = gaussian_velocity.get_variances()[pick]
			data = torch.distributions.MultivariateNormal(gaussian_velocity.positions[pick], precision_matrix=(variances + variances.transpose(-1, -2)) * .5).sample()
		data.clamp_(
			torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device),
			torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
		)
		return data
	
	fit_velocity_with_gradient(gaussian_velocity, velocity_ref, velocity_ref.gradient, default_generator, max_epoch=500)
	gaussian_velocity.save(os.path.join(cmd_args.dir, 'gaussian_velocity_0.pt'))
	
	def vorticity(x):
		g = gaussian_velocity.gradient(x)
		vor = torch.zeros((x.shape[0], 3), device=device)
		vor[:, 0] = g[:, 2, 1] - g[:, 1, 2]
		vor[:, 1] = g[:, 0, 2] - g[:, 2, 0]
		vor[:, 2] = g[:, 1, 0] - g[:, 0, 1]
		return vor
	def divergence(x):
		return gaussian_velocity.gradient(x).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
	
	write_vti(lambda x: (vorticity(x) ** 2).sum(axis=-1) ** .5, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'vorticity_0.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	write_vti(divergence, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, 'divergence_0.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)


if __name__ == '__main__':
	SimulationInitialize()
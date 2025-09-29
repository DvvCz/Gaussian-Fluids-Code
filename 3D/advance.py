import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from init_cond import *


class AdvectedCovectorField:
	def __init__(self, origin_covector_field, velocity_field, time_step, x_min, x_max, y_min, y_max, z_min, z_max, advection_scheme='rk4'):
		'''
		Advect origin_covector_field with velocity_field for time_step seconds.
		The fluid domain is [x_min, x_max] x [y_min, y_max] x [z_min, z_max].
		The advection scheme is defined by advection_scheme (can be 'rk4' or 'rk1-backtrace').
		'''
		self.origin_covector_field = origin_covector_field
		self.velocity_field = velocity_field
		self.time_step = time_step
		self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = x_min, x_max, y_min, y_max, z_min, z_max
		self.advection_scheme = advection_scheme
	
	def vorticity(self, x, need_hel=False):
		'''
		Evaluate the vorticities of the advected covector field at positions x.
		Might advect the positions x as well.
		Return: df_vor, the vorticity of the covector field.
		Warning: advection_scheme='rk4' would modify x to the advected positions and return the values at the new x.
		'''
		if self.advection_scheme == 'rk1-backtrace':
			raise NotImplementedError
		
		elif self.advection_scheme == 'rk4':
			psi, dpsi, pb_v, pb_dv = self.velocity_field.advection_rk4(x, -self.time_step, pos_only=False)
			
			def get_vor(jacob):
				vor = torch.zeros((x.shape[0], 3), device=device)
				vor[:, 0] = jacob[:, 2, 1] - jacob[:, 1, 2]
				vor[:, 1] = jacob[:, 0, 2] - jacob[:, 2, 0]
				vor[:, 2] = jacob[:, 1, 0] - jacob[:, 0, 1]
				return vor
			pb_vor = get_vor(pb_dv)
			if need_hel:
				hel = (pb_v * pb_vor).sum(axis=-1)
			vor = (dpsi.inverse() @ pb_vor.unsqueeze(-1)).squeeze(-1)
			return (vor, hel) if need_hel else vor
		
		raise NotImplementedError

def clone_velocity_field(res: GaussianSplatting3DFast, velocity_field: GaussianSplatting3DFast, x_min, x_max, y_min, y_max, z_min, z_max, data_generator, test_data_generator, reinitialize=False, batch_size=8192, max_epoch=3000, patience=500, verbose=1):
	# Reseeding
	with torch.no_grad():
		res.positions = velocity_field.positions.clone()
		res.scalings = velocity_field.scalings.clone()
		res.rotations = velocity_field.rotations.clone()
		res.values = velocity_field.values.clone()
		res.N = res.positions.shape[0]
		
		# Splitting points
		stop_gradient = torch.ones((res.N,), dtype=torch.bool, device=device)
		while True:
			min_scalings, split_axises = res.scalings.min(axis=-1)
			scalings_ratio = torch.exp(-min_scalings + res.scalings.max(axis=-1).values)
			need_split = scalings_ratio >= 2.
			print(f'Add {need_split.sum()} particles. {scalings_ratio.max()}')
			if need_split.any():
				split_variances = res.get_variances()[need_split]
				split_positions = torch.distributions.MultivariateNormal(res.positions[need_split], precision_matrix=(split_variances + split_variances.transpose(-1, -2)) * .5).sample((2,)).flatten(0, 1)
				split_positions.clamp_(
					torch.tensor([res.x_min, res.y_min, res.z_min], dtype=torch.float32, device=device),
					torch.tensor([res.x_max, res.y_max, res.z_max], dtype=torch.float32, device=device)
				)
				split_rotations = res.rotations[need_split].repeat(2, 1)
				res.scalings[need_split, split_axises[need_split]] += np.log(2.)
				res.scalings[need_split] -= np.log(2.) / 3.
				split_scalings = res.scalings[need_split].repeat(2, 1)
				split_values = res.values[need_split].repeat(2, 1)
				
				res.positions = torch.cat([res.positions[~need_split], split_positions], dim=0)
				res.rotations = torch.cat([res.rotations[~need_split], split_rotations], dim=0)
				res.scalings = torch.cat([res.scalings[~need_split], split_scalings], dim=0)
				res.values = torch.cat([res.values[~need_split], split_values], dim=0)
				res.N = res.positions.shape[0]
				stop_gradient = torch.cat([stop_gradient[~need_split], torch.zeros((split_positions.shape[0],), dtype=torch.bool, device=device)], dim=0)
			else:
				break
	res.unfreeze()
	res.zero_grad()
	
	if stop_gradient.all():
		return res
	stop_gradient = torch.logical_and(stop_gradient, ~res.get_all_neighbors(res.positions[~stop_gradient].detach()))
	
	# Training for res to represent velocity_field
	def get_losses(data, backward=True):
		weight_val, weight_grad, weight_aniso, weight_vol = 1., 1., 1., 1.
		
		ref_val, ref_grad = velocity_field.get_losses(data)
		if backward:
			val, grad = res.get_losses(data, ref_val=ref_val, weight_val=weight_val, ref_grad=ref_grad, weight_grad=weight_grad, stop_gradient=stop_gradient)
		else:
			val, grad = res.get_losses(data)
		loss_val = F.l1_loss(val, ref_val)
		loss_grad = F.l1_loss(grad, ref_grad)
		
		aniso_ratio = 1.5
		scalings_ratio = torch.exp(-res.scalings[~stop_gradient].min(axis=-1).values + res.scalings[~stop_gradient].max(axis=-1).values)
		if not scalings_ratio.shape[0]:
			scalings_ratio = torch.ones((1,), device=device)
		loss_aniso = (torch.where(scalings_ratio >= aniso_ratio, scalings_ratio, aniso_ratio) - aniso_ratio).mean()
		volumes = torch.zeros((res.scalings.shape[0],), device=device)
		volumes[stop_gradient] = torch.exp(-res.scalings.detach()[stop_gradient].sum(axis=-1))
		volumes[~stop_gradient] = torch.exp(-res.scalings[~stop_gradient].sum(axis=-1))
		loss_vol = ((volumes / volumes.mean() - 1) ** 2).mean()
		if backward:
			(weight_aniso * loss_aniso + weight_vol * loss_vol).backward()
		
		return weight_val * loss_val + weight_grad * loss_grad + weight_aniso * loss_aniso + weight_vol * loss_vol, loss_val, loss_grad, loss_aniso, loss_vol
	
	res.positions_lr = 1e-3
	res.rotations_lr = 1e-3
	res.scalings_lr = 1e-3
	res.values_lr = 1e-3
	res.initialize_optimizers()
	res.positions_scheduler.factor = .9
	res.rotations_scheduler.factor = .9
	res.scalings_scheduler.factor = .9
	res.values_scheduler.factor = .9
	
	_, loss_val, loss_grad, loss_aniso, loss_vol = get_losses(test_data, backward=False)
	if verbose:
		print(f'[clone] loss: {loss_val.item()}, loss_grad: {loss_grad.item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}')
	if stop_gradient.all():
		return res
	
	st_time = time.time()
	check_iter = 100
	min_loss_val, min_loss_grad, iter_loss_val, iter_loss_grad = np.inf, np.inf, 0, 0
	for epoch in range(max_epoch):
		data = data_generator(batch_size, res, ~stop_gradient)
		loss_tot, loss_val, loss_grad, loss_aniso, loss_vol = get_losses(data)
		res.step(loss_tot)
		
		if epoch % check_iter == check_iter - 1:
			test_data = test_data_generator(res)
			_, loss_val, loss_grad, loss_aniso, loss_vol = get_losses(test_data, backward=False)
			if loss_val.item() < min_loss_val * (1. - 1e-3):
				min_loss_val, iter_loss_val = loss_val.item(), 0
			else:
				iter_loss_val += check_iter
			if loss_grad.item() < min_loss_grad * (1. - 1e-3):
				min_loss_grad, iter_loss_grad = loss_grad.item(), 0
			else:
				iter_loss_grad += check_iter
			if verbose:
				print(f'[clone] loss: {loss_val.item()}, loss_grad: {loss_grad.item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, time: {time.time() - st_time}')
				st_time = time.time()
			if iter_loss_val >= patience and iter_loss_grad >= patience:
				print(f'[clone] Total epoch:', epoch + 1)
				break
	else:
		print(f'[clone] Total epoch:', max_epoch, '(Reached maximum iteration number)')
	
	return res

def advect_covector_field(covector_field: GaussianSplatting3DFast, velocity_field: GaussianSplatting3DFast, dt, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None, advection_scheme='rk4'):
	# advect positions
	if advection_scheme == 'rk1-backtrace':
		raise NotImplementedError
	elif advection_scheme == 'rk4':
		new_positions = velocity_field.advection_rk4(covector_field.positions, dt)
	else:
		raise NotImplementedError
	if x_min is not None:
		new_positions.clamp_(torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device), torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device))
	
	new_positions.requires_grad_()
	covector_field.positions = new_positions
	covector_field.zero_grad()

def project(gaussian_velocity: GaussianSplatting3DFast, reference_field: AdvectedCovectorField, x_min, x_max, y_min, y_max, z_min, z_max, data_generator, test_data_generator, boundary_generator=None, boundary_lambda=0., batch_size=8192, max_epoch=3000, patience=500, verbose=1, frame_id=None):
	def get_losses(data, backward=True):
		weight_vor, weight_hel, weight_div, weight_aniso, weight_vol, weight_val_reg = 1., 1., 1., 10., 10., .0
		
		ref_vor, ref_hel = reference_field.vorticity(data, need_hel=True)	# reference_field may change data, must execute first
		if backward:
			vor_positions_grad = torch.zeros_like(gaussian_velocity.positions, device=device)
			vor_scalings_grad = torch.zeros_like(gaussian_velocity.scalings, device=device)
			vor_rotations_grad = torch.zeros_like(gaussian_velocity.rotations, device=device)
			vor_values_grad = torch.zeros_like(gaussian_velocity.values, device=device)
			div_positions_grad = torch.zeros_like(gaussian_velocity.positions, device=device)
			div_scalings_grad = torch.zeros_like(gaussian_velocity.scalings, device=device)
			div_rotations_grad = torch.zeros_like(gaussian_velocity.rotations, device=device)
			div_values_grad = torch.zeros_like(gaussian_velocity.values, device=device)
			val, grad = gaussian_velocity.get_losses(
				data, ref_vor=ref_vor, weight_vor=weight_vor, ref_hel=ref_hel, weight_hel=weight_hel,
				weight_div=weight_div,
				vor_positions_grad=vor_positions_grad, vor_scalings_grad=vor_scalings_grad, vor_rotations_grad=vor_rotations_grad, vor_values_grad=vor_values_grad,
				div_positions_grad=div_positions_grad, div_scalings_grad=div_scalings_grad, div_rotations_grad=div_rotations_grad, div_values_grad=div_values_grad
			)
			if (vor_positions_grad * div_positions_grad).sum() < 0.:
				n1 = vor_positions_grad / (vor_positions_grad ** 2).sum() ** .5
				n2 = div_positions_grad / (div_positions_grad ** 2).sum() ** .5
				vor_positions_grad -= (vor_positions_grad * n2).sum() * n2
				div_positions_grad -= (div_positions_grad * n1).sum() * n1
			gaussian_velocity.positions.grad += vor_positions_grad + div_positions_grad
			if (vor_scalings_grad * div_scalings_grad).sum() < 0.:
				n1 = vor_scalings_grad / (vor_scalings_grad ** 2).sum() ** .5
				n2 = div_scalings_grad / (div_scalings_grad ** 2).sum() ** .5
				vor_scalings_grad -= (vor_scalings_grad * n2).sum() * n2
				div_scalings_grad -= (div_scalings_grad * n1).sum() * n1
			gaussian_velocity.scalings.grad += vor_scalings_grad + div_scalings_grad
			if (vor_rotations_grad * div_rotations_grad).sum() < 0.:
				n1 = vor_rotations_grad / (vor_rotations_grad ** 2).sum() ** .5
				n2 = div_rotations_grad / (div_rotations_grad ** 2).sum() ** .5
				vor_rotations_grad -= (vor_rotations_grad * n2).sum() * n2
				div_rotations_grad -= (div_rotations_grad * n1).sum() * n1
			gaussian_velocity.rotations.grad += vor_rotations_grad + div_rotations_grad
			if (vor_values_grad * div_values_grad).sum() < 0.:
				n1 = vor_values_grad / (vor_values_grad ** 2).sum() ** .5
				n2 = div_values_grad / (div_values_grad ** 2).sum() ** .5
				vor_values_grad -= (vor_values_grad * n2).sum() * n2
				div_values_grad -= (div_values_grad * n1).sum() * n1
			gaussian_velocity.values.grad += vor_values_grad + div_values_grad
		else:
			grad, val = gaussian_velocity.gradient(data, need_val=True)
		vor = torch.zeros((data.shape[0], 3), device=device)
		vor[:, 0] = grad[:, 2, 1] - grad[:, 1, 2]
		vor[:, 1] = grad[:, 0, 2] - grad[:, 2, 0]
		vor[:, 2] = grad[:, 1, 0] - grad[:, 0, 1]
		loss_vor = torch.abs(vor - ref_vor).mean(axis=-1)
		loss_hel = torch.abs((val * vor).sum(axis=-1) - ref_hel)
		# loss_div = torch.abs(grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2])
		loss_div = (grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2]) ** 2
		
		aniso_ratio = 1.5
		scalings_ratio = torch.exp(-gaussian_velocity.scalings.min(axis=-1).values + gaussian_velocity.scalings.max(axis=-1).values)
		loss_aniso = (torch.where(scalings_ratio >= aniso_ratio, scalings_ratio, aniso_ratio) - aniso_ratio).mean()
		volumes = torch.exp(-gaussian_velocity.scalings.sum(axis=-1))
		loss_vol = ((volumes / volumes.mean() - 1) ** 2).mean()
		loss_val_reg = gaussian_velocity.values.abs().mean()
		if backward:
			(weight_aniso * loss_aniso + weight_vol * loss_vol + weight_val_reg * loss_val_reg).backward()
		
		boundary_constraint = torch.tensor(0., device=device)
		if boundary_lambda and boundary_generator:
			boundary_data, boundary_normal = boundary_generator(batch_size)
			if backward:
				boundary_output, _ = gaussian_velocity.get_losses(boundary_data, normals=boundary_normal, weight_boundary=boundary_lambda)
			else:
				boundary_output = gaussian_velocity(boundary_data)
			boundary_constraint = (boundary_output * boundary_normal).sum(axis=1).abs().mean()
		
		loss_tot = weight_vor * loss_vor.mean() + weight_div * loss_div.mean() + weight_aniso * loss_aniso + weight_vol * loss_vol + weight_val_reg * loss_val_reg + boundary_lambda * boundary_constraint
		return loss_tot, loss_vor, loss_hel, loss_div, loss_aniso, loss_vol, loss_val_reg, boundary_constraint
	
	gaussian_velocity.positions_lr = 3e-4
	gaussian_velocity.scalings_lr = 1e-5
	gaussian_velocity.rotations_lr = 3e-4
	gaussian_velocity.values_lr = 1e-5
	gaussian_velocity.initialize_optimizers(patience=50)
	gaussian_velocity.positions_scheduler.factor = .9
	gaussian_velocity.scalings_scheduler.factor = .9
	gaussian_velocity.rotations_scheduler.factor = .9
	gaussian_velocity.values_scheduler.factor = .9
	
	if verbose:
		test_data = test_data_generator(gaussian_velocity)
		_, loss_vor, loss_hel, loss_div, loss_aniso, loss_vol, loss_val_reg, boundary_constraint = get_losses(test_data, backward=False)
		print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_hel: {loss_hel.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_val_reg: {loss_val_reg.item()}, boundary_constraint: {boundary_constraint.item()}')
	
	if frame_id is not None:
		train_loss_vor, train_loss_div = [], []
		log_learning_rate = []
		test_loss_vor, test_loss_div = [], []
	st_time = time.time()
	check_iter = 100
	min_loss_vor, min_loss_hel, iter_loss_vor, iter_loss_hel = np.inf, np.inf, 0, 0
	min_loss_div, iter_loss_div = np.inf, 0
	for epoch in range(max_epoch):
		data = data_generator(batch_size, gaussian_velocity)
		loss_tot, loss_vor, loss_hel, loss_div, loss_aniso, loss_vol, loss_val_reg, boundary_constraint = get_losses(data)
		gaussian_velocity.step(loss_tot)
		train_loss_vor.append(loss_vor.mean().item())
		train_loss_div.append(loss_div.mean().item())
		log_learning_rate.append(np.log(gaussian_velocity.scalings_optimizer.state_dict()['param_groups'][0]['lr']))
		
		if epoch % check_iter == check_iter - 1:
			test_data = test_data_generator(gaussian_velocity)
			_, loss_vor, loss_hel, loss_div, loss_aniso, loss_vol, loss_val_reg, boundary_constraint = get_losses(test_data, backward=False)
			test_loss_vor.append(loss_vor.mean().item())
			test_loss_div.append(loss_div.mean().item())
			if verbose:
				print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_hel: {loss_hel.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_val_reg: {loss_val_reg.item()}, boundary_constraint: {boundary_constraint.item()}, time: {time.time() - st_time}')
				st_time = time.time()
			if loss_vor.mean().item() < min_loss_vor * (1. - 1e-3):
				iter_loss_vor = 0
				min_loss_vor = loss_vor.mean().item()
			else:
				iter_loss_vor += check_iter
			if loss_hel.mean().item() < min_loss_hel * (1. - 1e-3):
				iter_loss_hel = 0
				min_loss_hel = loss_hel.mean().item()
			else:
				iter_loss_hel += check_iter
			if loss_div.mean().item() < min_loss_div * (1. - 1e-3):
				iter_loss_div = 0
				min_loss_div = loss_div.mean().item()
			else:
				iter_loss_div += check_iter
			if iter_loss_vor >= patience and iter_loss_hel >= patience and iter_loss_div >= patience:
				print('[projection] Total epoch:', epoch + 1)
				break
	else:
		print('[projection] Total epoch:', max_epoch, '(Reached maximum iteration number)')
	if frame_id is not None:
		_, axs = plt.subplots(2, 2, figsize=(12, 10))
		axs[0, 0].plot(list(range(len(train_loss_vor))), train_loss_vor)
		tmp_ax = axs[0, 0].twinx()
		tmp_ax.plot(list(range(len(train_loss_vor))), log_learning_rate, color='orange')
		axs[0, 0].set_title('Vorticity training loss')
		axs[0, 1].plot(list(range(len(train_loss_div))), train_loss_div)
		axs[0, 1].set_title('Divergence training loss')
		axs[1, 0].plot(list(range(len(test_loss_vor))), test_loss_vor)
		axs[1, 0].set_title('Vorticity test loss')
		axs[1, 1].plot(list(range(len(test_loss_div))), test_loss_div)
		axs[1, 1].set_title('Divergence test loss')
		plt.tight_layout()
		plt.savefig(os.path.join(cmd_args.dir, f'loss_{frame_id}.png'))
		plt.clf()


if __name__ == '__main__':
	
	x_min, x_max, y_min, y_max, z_min, z_max = domain[cmd_args.init_cond]
	x_Nvis, y_Nvis, z_Nvis = visualize_res[cmd_args.init_cond]
	
	def default_data_generator(n, gaussian_splatting: GaussianSplatting3DFast, restrict=None):
		return torch.rand_like(gaussian_splatting.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=device) + torch.tensor([x_min, y_min, z_min], device=device)
	def default_test_generator(gaussian_splatting: GaussianSplatting3DFast):
		return get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_Nvis, y_Nvis, z_Nvis)
	def generate_gaussians(n, gaussian_splatting: GaussianSplatting3DFast, restrict=None):
		if restrict is None:
			restrict = torch.ones((gaussian_splatting.N,), dtype=torch.bool, device=device)
		with torch.no_grad():
			positions = gaussian_splatting.positions[restrict]
			variances = gaussian_splatting.get_variances()[restrict]
			N = positions.shape[0]
			pick = torch.randint(0, N, (n,), device=device)
			data = torch.distributions.MultivariateNormal(positions[pick], precision_matrix=(variances[pick] + variances[pick].transpose(-1, -2)) * .5).sample().clamp(
				torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device),
				torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
			)
		return data
	def generate_all_gaussians(gaussian_splatting: GaussianSplatting3DFast):
		with torch.no_grad():
			variances = gaussian_splatting.get_variances()
			data = torch.distributions.MultivariateNormal(gaussian_splatting.positions, precision_matrix=(variances + variances.transpose(-1, -2)) * .5).sample().clamp(
				torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device),
				torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
			)
		return data
	
	gaussian_velocity = GaussianSplatting3DFast(x_min, x_max, y_min, y_max, z_min, z_max, np.zeros((1, 3)), dim=3, load_file=os.path.join(cmd_args.dir, f'gaussian_velocity_{cmd_args.start_frame}.pt'))
	new_gaussian_velocity = GaussianSplatting3DFast(x_min, x_max, y_min, y_max, z_min, z_max, np.zeros((1, 3)), dim=3, load_file=os.path.join(cmd_args.dir, f'gaussian_velocity_{cmd_args.start_frame}.pt'))
	
	def vorticity(x):
		g = gaussian_velocity.gradient(x)
		vor = torch.zeros((x.shape[0], 3), device=device)
		vor[:, 0] = g[:, 2, 1] - g[:, 1, 2]
		vor[:, 1] = g[:, 0, 2] - g[:, 2, 0]
		vor[:, 2] = g[:, 1, 0] - g[:, 0, 1]
		return vor
	def divergence(x):
		return gaussian_velocity.gradient(x).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
	
	write_vti(lambda x: (vorticity(x) ** 2).sum(axis=-1) ** .5, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'vorticity_{cmd_args.start_frame}.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	write_vti(divergence, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'divergence_{cmd_args.start_frame}.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
	
	t, time_step = 0, cmd_args.dt
	cnt = cmd_args.start_frame + 1
	while t < cmd_args.last_time:
		clone_velocity_field(new_gaussian_velocity, gaussian_velocity, x_min, x_max, y_min, y_max, z_min, z_max, default_data_generator, default_test_generator, reinitialize=False, max_epoch=20000, verbose=1)
		advect_covector_field(new_gaussian_velocity, gaussian_velocity, time_step, new_gaussian_velocity.x_min, new_gaussian_velocity.x_max, new_gaussian_velocity.y_min, new_gaussian_velocity.y_max, new_gaussian_velocity.z_min, new_gaussian_velocity.z_max)
		project(new_gaussian_velocity, AdvectedCovectorField(gaussian_velocity, gaussian_velocity, time_step, x_min, x_max, y_min, y_max, z_min, z_max), x_min, x_max, y_min, y_max, z_min, z_max, default_data_generator, default_test_generator, boundary_lambda=cmd_args.boundary, boundary_generator=boundary_sampler[cmd_args.init_cond], max_epoch=20000, verbose=1, frame_id=cnt)
		gaussian_velocity, new_gaussian_velocity = new_gaussian_velocity, gaussian_velocity
		print(f"Wrote frame {cnt}")
		write_vti(lambda x: (vorticity(x) ** 2).sum(axis=-1) ** .5, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'vorticity_{cnt}.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
		write_vti(divergence, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'divergence_{cnt}.vti'), x_N=x_Nvis, y_N=y_Nvis, z_N=z_Nvis)
		gaussian_velocity.save(os.path.join(cmd_args.dir, f'gaussian_velocity_{cnt}.pt'))
		cnt += 1
		t += time_step
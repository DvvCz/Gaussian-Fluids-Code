import torch
import torch.nn.functional as F
import numpy as np
import time

from init_cond import *


class AdvectedCovectorField:
	def __init__(self, origin_covector_field, velocity_field, time_step, advection_scheme='rk4'):
		'''
		Advect origin_covector_field with velocity_field for time_step seconds.
		The fluid domain is [x_min, x_max] x [y_min, y_max].
		The advection scheme is defined by advection_scheme (can be 'rk4' or 'rk1-backtrace').
		'''
		self.origin_covector_field = origin_covector_field
		self.velocity_field = velocity_field
		self.time_step = time_step
		self.advection_scheme = advection_scheme
	
	def vorticity(self, x):
		'''
		Evaluate the vorticities of the advected covector field at positions x.
		Might advect the positions x as well.
		Return: df_vor, the vorticity of the covector field.
		Warning: advection_scheme='rk4' would modify x to the advected positions and return the values at the new x.
		'''
		x_min, x_max, y_min, y_max = advance_domain[cmd_args.init_cond]
		x_min *= scaling_factor
		x_max *= scaling_factor
		y_min *= scaling_factor
		y_max *= scaling_factor
		
		if self.advection_scheme == 'rk1-backtrace':
			with torch.no_grad():
				x_backtrace = x - self.velocity_field(x) * self.time_step
				deformation = torch.eye(2, device=device)[None] - self.time_step * self.velocity_field.gradient(x)
				dv = self.velocity_field.gradient(x_backtrace)
				vor = dv[:, 1, 0] - dv[:, 0, 1]
				# v = (deformation.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
				
				out_of_range = torch.logical_or(torch.logical_or(x_backtrace[:, 0] < x_min, x_backtrace[:, 0] > x_max), torch.logical_or(x_backtrace[:, 1] < y_min, x_backtrace[:, 1] > y_max))
				vor[out_of_range] = 0.
			return vor
		
		elif self.advection_scheme == 'rk4':
			with torch.no_grad():
				bk_x, deformation, v, dv = self.velocity_field.advection_rk4(x, -self.time_step, pos_only=False)
				vor = dv[:, 1, 0] - dv[:, 0, 1]
				# v = (deformation.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
				
				out_of_range = torch.logical_or(torch.logical_or(bk_x[:, 0] < x_min, bk_x[:, 0] > x_max), torch.logical_or(bk_x[:, 1] < y_min, bk_x[:, 1] > y_max))
				vor[out_of_range] = 0.
			return vor
		
		raise NotImplementedError

def clone_velocity_field(res: GaussianSplattingFast, velocity_field: GaussianSplattingFast, data_generator, test_data_generator, batch_size=512, max_epoch=3000, patience=500, verbose=1):
	# Reseeding
	with torch.no_grad():
		res.positions = velocity_field.positions.clone()
		res.scalings = velocity_field.scalings.clone()
		res.rotations = velocity_field.rotations.clone()
		res.values = velocity_field.values.clone()
		res.N = res.positions.shape[0]
		
		# Splitting points
		scalings_ratio = torch.exp(-res.scalings.min(axis=-1).values + res.scalings.max(axis=-1).values)
		need_split = scalings_ratio >= 1.5
		if need_split.any():
			split_variances = res.get_variances()[need_split]
			split_positions = torch.distributions.MultivariateNormal(res.positions[need_split], precision_matrix=(split_variances + split_variances.transpose(-1, -2)) * .5).sample((2,)).flatten(0, 1)
			split_rotations = res.rotations[need_split].repeat(2)
			split_scalings = res.scalings[need_split].repeat(2, 1)
			split_axis_1 = split_scalings[:, 1] < split_scalings[:, 0]
			split_scalings[split_axis_1, 1] += np.log(1.5)
			split_scalings[~split_axis_1, 0] += np.log(1.5)
			split_values = res.values[need_split].repeat(2, 1)
			
			res.positions = torch.cat([res.positions[~need_split], split_positions], dim=0)
			res.rotations = torch.cat([res.rotations[~need_split], split_rotations], dim=0)
			res.scalings = torch.cat([res.scalings[~need_split], split_scalings], dim=0)
			res.values = torch.cat([res.values[~need_split], split_values], dim=0)
			res.N = res.positions.shape[0]
		stop_gradient = torch.zeros((res.N,), dtype=torch.bool, device=device)
		stop_gradient[:(~need_split).sum().item()] = True
	res.unfreeze()
	res.zero_grad()
	
	if stop_gradient.all():
		return res
	stop_gradient = torch.logical_and(stop_gradient, ~res.get_all_neighbors(res.positions[~stop_gradient].detach()))
	print(f'[clone] Add {need_split.sum().item()} particles.')
	
	# Training for res to represent velocity_field
	def get_losses(data, backward=True):
		weight, weight_grad, weight_aniso, weight_vol = 1., 1., 1., 1.
		
		ref_grad, ref_val = velocity_field.gradient(data, need_val=True)
		if backward:
			val = res.get_losses(data, ref=ref_val, weight=weight, stop_gradient=stop_gradient.int())
			grad = res.get_grad_losses(data, ref_grad=ref_grad, weight_grad=weight_grad, stop_gradient=stop_gradient.int())
		else:
			grad, val = res.gradient(data, need_val=True)
		loss = F.l1_loss(val, ref_val)
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
		
		return weight * loss + weight_grad * loss_grad + weight_aniso * loss_aniso + weight_vol * loss_vol, loss, loss_grad, loss_aniso, loss_vol
	
	res.set_lr(positions_lr=1e-2, rotations_lr=5e-2, scalings_lr=5e-2, values_lr=5e-3)
	res.initialize_optimizers()
	
	test_data = test_data_generator(res)
	_, loss, loss_grad, loss_aniso, loss_vol = get_losses(test_data, backward=False)
	if verbose:
		print(f'[clone] loss: {loss.item()}, loss_grad: {loss_grad.item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}')
	
	st_time = time.time()
	check_iter = 100
	min_loss, min_loss_grad, iter_loss, iter_loss_grad = np.inf, np.inf, 0, 0
	for epoch in range(max_epoch):
		data = data_generator(batch_size, res, ~stop_gradient)
		loss_tot, loss, loss_grad, loss_aniso, loss_vol = get_losses(data)
		res.step(loss_tot)
		
		if epoch % check_iter == check_iter - 1:
			test_data = test_data_generator(res)
			_, loss, loss_grad, loss_aniso, loss_vol = get_losses(test_data, backward=False)
			if loss.item() < min_loss * (1. - 1e-3):
				min_loss, iter_loss = loss.item(), 0
			else:
				iter_loss += check_iter
			if loss_grad.item() < min_loss_grad * (1. - 1e-3):
				min_loss_grad, iter_loss_grad = loss_grad.item(), 0
			else:
				iter_loss_grad += check_iter
			if verbose:
				print(f'[clone] loss: {loss.item()}, loss_grad: {loss_grad.item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, time: {time.time() - st_time}')
				st_time = time.time()
			if iter_loss >= patience and iter_loss_grad >= patience:
				print(f'[clone] Total epoch:', epoch + 1)
				break
	else:
		print(f'[clone] Total epoch:', max_epoch, '(Reached maximum iteration number)')
	
	return res

def advect_covector_field(covector_field: GaussianSplattingFast, velocity_field: GaussianSplattingFast, dt, advection_scheme='rk4'):
	with torch.no_grad():
		# advect positions
		if advection_scheme == 'rk1-backtrace':
			new_positions = covector_field.positions + dt * velocity_field(covector_field.positions)
		elif advection_scheme == 'rk4':
			new_positions = covector_field.advection_rk4(covector_field.positions, dt)
		else:
			raise NotImplementedError
		
		valid_points = torch.logical_and(torch.logical_and(covector_field.x_min <= new_positions[:, 0], new_positions[:, 0] <= covector_field.x_max), torch.logical_and(covector_field.y_min <= new_positions[:, 1], new_positions[:, 1] <= covector_field.y_max))
		new_positions = new_positions[valid_points, :].clone()
		new_scalings = covector_field.scalings[valid_points, :].clone()
		new_rotations = covector_field.rotations[valid_points].clone()
		new_values = covector_field.values[valid_points, :].clone()
	
	new_positions.requires_grad_()
	new_scalings.requires_grad_()
	new_rotations.requires_grad_()
	new_values.requires_grad_()
	covector_field.positions, covector_field.scalings, covector_field.rotations, covector_field.values = new_positions, new_scalings, new_rotations, new_values
	covector_field.N = covector_field.positions.shape[0]
	covector_field.zero_grad()
	
	if extra_advector[cmd_args.init_cond]:
		extra_advector[cmd_args.init_cond](dt, advection_scheme)

def project(gaussian_velocity: GaussianSplattingFast, reference_field: AdvectedCovectorField, data_generator, test_data_generator, boundary_generator_1=None, boundary_generator_2=None, boundary_lambda=0., batch_size=512, max_epoch=3000, patience=500, verbose=1):
	def gradient_project(g1, g2):
		if (g1 * g2).sum() < 0.:
			n1 = g1 / (g1 ** 2).sum() ** .5
			n2 = g2 / (g2 ** 2).sum() ** .5
			g1 -= (g1 * n2).sum() * n2
			g2 -= (g2 * n1).sum() * n1
	
	positions_org = gaussian_velocity.positions.detach().clone()
	
	def get_losses(data, backward=True):
		weight_vor, weight_div, weight_aniso, weight_vol, weight_delta_pos = 1., 1., 10., 10., .5
		
		ref_vor = reference_field.vorticity(data)	# reference_field may change data, must execute first
		boundary_constraint = torch.tensor(0., device=device)
		if backward:
			vor_positions_grad = torch.zeros_like(gaussian_velocity.positions, device=device)
			vor_scalings_grad = torch.zeros_like(gaussian_velocity.scalings, device=device)
			vor_rotations_grad = torch.zeros_like(gaussian_velocity.rotations, device=device)
			vor_values_grad = torch.zeros_like(gaussian_velocity.values, device=device)
			div_positions_grad = torch.zeros_like(gaussian_velocity.positions, device=device)
			div_scalings_grad = torch.zeros_like(gaussian_velocity.scalings, device=device)
			div_rotations_grad = torch.zeros_like(gaussian_velocity.rotations, device=device)
			div_values_grad = torch.zeros_like(gaussian_velocity.values, device=device)
			grad = gaussian_velocity.get_grad_losses(
				data, ref_vor=ref_vor, weight_vor=weight_vor, weight_div=weight_div,
				vor_positions_grad=vor_positions_grad, vor_scalings_grad=vor_scalings_grad, vor_rotations_grad=vor_rotations_grad, vor_values_grad=vor_values_grad,
				div_positions_grad=div_positions_grad, div_scalings_grad=div_scalings_grad, div_rotations_grad=div_rotations_grad, div_values_grad=div_values_grad
			)
			
			if boundary_lambda > 0. and boundary_generator_1:
				boundary_data, boundary_value = boundary_generator_1(batch_size)
				boundary_output = gaussian_velocity.get_losses(boundary_data, ref=boundary_value, weight=boundary_lambda)
				boundary_constraint += F.l1_loss(boundary_output, boundary_value)
			
			gradient_project(vor_positions_grad, div_positions_grad)
			gaussian_velocity.positions.grad += vor_positions_grad + div_positions_grad
			gradient_project(vor_scalings_grad, div_scalings_grad)
			gaussian_velocity.scalings.grad += vor_scalings_grad + div_scalings_grad
			gradient_project(vor_rotations_grad, div_rotations_grad)
			gaussian_velocity.rotations.grad += vor_rotations_grad + div_rotations_grad
			gradient_project(vor_values_grad, div_values_grad)
			gaussian_velocity.values.grad += vor_values_grad + div_values_grad
			
			if boundary_lambda > 0. and boundary_generator_2:
				boundary_data, boundary_normal, boundary_normal_ref = boundary_generator_2(batch_size)
				boundary_output = gaussian_velocity.get_losses(boundary_data, normals=boundary_normal, normal_ref=boundary_normal_ref, weight_boundary=boundary_lambda)
				boundary_constraints = (boundary_output * boundary_normal).sum(axis=1)
				boundary_constraint += F.l1_loss(boundary_constraints, boundary_normal_ref)
		else:
			grad, val = gaussian_velocity.gradient(data, need_val=True)
			if boundary_lambda > 0. and boundary_generator_1:
				boundary_data, boundary_value = boundary_generator_1(batch_size)
				boundary_output = gaussian_velocity(boundary_data)
				boundary_constraint += F.l1_loss(boundary_output, boundary_value)
			if boundary_lambda > 0. and boundary_generator_2:
				boundary_data, boundary_normal, boundary_normal_ref = boundary_generator_2(batch_size)
				boundary_output = gaussian_velocity(boundary_data)
				boundary_constraints = (boundary_output * boundary_normal).sum(axis=1)
				boundary_constraint += F.l1_loss(boundary_constraints, boundary_normal_ref)
		loss_vor = torch.abs(grad[:, 1, 0] - grad[:, 0, 1] - ref_vor)
		loss_div = (grad[:, 0, 0] + grad[:, 1, 1]) ** 2
		
		aniso_ratio = 1.5
		scalings_ratio = torch.exp(-gaussian_velocity.scalings.min(axis=-1).values + gaussian_velocity.scalings.max(axis=-1).values)
		loss_aniso = (torch.where(scalings_ratio >= aniso_ratio, scalings_ratio, aniso_ratio) - aniso_ratio).mean()
		volumes = torch.exp(-gaussian_velocity.scalings.sum(axis=-1))
		loss_vol = ((volumes / volumes.mean() - 1) ** 2).mean()
		loss_delta_pos = F.mse_loss(gaussian_velocity.positions, positions_org)
		if backward:
			(weight_aniso * loss_aniso + weight_vol * loss_vol + weight_delta_pos * loss_delta_pos).backward()
		
		return weight_vor * loss_vor.mean() + weight_div * loss_div.mean() + weight_aniso * loss_aniso + weight_vol * loss_vol + weight_delta_pos * loss_delta_pos + boundary_lambda * boundary_constraint, loss_vor, loss_div, loss_aniso, loss_vol, loss_delta_pos, boundary_constraint
	
	gaussian_velocity.set_lr(positions_lr=1e-4, scalings_lr=1e-4, rotations_lr=1e-4, values_lr=1e-4)
	gaussian_velocity.initialize_optimizers(patience=50)
	gaussian_velocity.positions_scheduler.factor = .9
	gaussian_velocity.scalings_scheduler.factor = .9
	gaussian_velocity.rotations_scheduler.factor = .9
	gaussian_velocity.values_scheduler.factor = .9
	
	test_data = test_data_generator(gaussian_velocity)
	if verbose:
		_, loss_vor, loss_div, loss_aniso, loss_vol, loss_delta_pos, boundary_constraint = get_losses(test_data, backward=False)
		print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_delta_pos: {loss_delta_pos.item()}, boundary_constraint: {boundary_constraint.item()}')
	
	st_time = time.time()
	check_iter = 100
	min_loss_vor, iter_loss_vor = np.inf, 0
	min_loss_div, iter_loss_div = np.inf, 0
	for epoch in range(max_epoch):
		data = data_generator(batch_size, gaussian_velocity)
		loss_tot, loss_vor, loss_div, loss_aniso, loss_vol, loss_delta_pos, boundary_constraint = get_losses(data)
		gaussian_velocity.step(loss_tot)
		
		if epoch % check_iter == check_iter - 1:
			test_data = test_data_generator(gaussian_velocity)
			_, loss_vor, loss_div, loss_aniso, loss_vol, loss_delta_pos, boundary_constraint = get_losses(test_data, backward=False)
			if verbose:
				print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_delta_pos: {loss_delta_pos.item()}, boundary_constraint: {boundary_constraint.item()}, time: {time.time() - st_time}')
				st_time = time.time()
			if loss_vor.mean().item() < min_loss_vor * (1. - 1e-3):
				iter_loss_vor = 0
				min_loss_vor = loss_vor.mean().item()
			else:
				iter_loss_vor += check_iter
			if loss_div.mean().item() < min_loss_div * (1. - 1e-2):
				iter_loss_div = 0
				min_loss_div = loss_div.mean().item()
			else:
				iter_loss_div += check_iter
			if iter_loss_vor >= patience and iter_loss_div >= patience:
				print('[projection] Total epoch:', epoch + 1)
				break
	else:
		print('[projection] Total epoch:', max_epoch, '(Reached maximum iteration number)')

if __name__ == '__main__':
	
	x_min_v, x_max_v, y_min_v, y_max_v = visualize_domain[cmd_args.init_cond]
	x_min_i, x_max_i, y_min_i, y_max_i = initialize_domain[cmd_args.init_cond]
	x_Nvis, y_Nvis = visualize_res[cmd_args.init_cond]
	x_min_i_gs, x_max_i_gs, y_min_i_gs, y_max_i_gs = x_min_i * scaling_factor, x_max_i * scaling_factor, y_min_i * scaling_factor, y_max_i * scaling_factor
	
	if extra_loader[cmd_args.init_cond]:
		extra_loader[cmd_args.init_cond]()
	
	def default_data_generator(n, gaussian_splatting: GaussianSplattingFast, restrict=None):
		x_min, x_max, y_min, y_max = advance_domain[cmd_args.init_cond]
		return (torch.rand_like(gaussian_splatting.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min], device=device) + torch.tensor([x_min, y_min], device=device)) * scaling_factor
	def default_test_generator(gaussian_splatting: GaussianSplattingFast):
		x_min, x_max, y_min, y_max = advance_domain[cmd_args.init_cond]
		return get_grid_points(x_min, x_max, y_min, y_max, x_Nvis, y_Nvis) * scaling_factor
	
	# lr_params = {
	# 	'positions_lr': 0.0001,
	# 	'scalings_lr': 0.003,
	# 	'rotations_lr': 7e-06,
	# 	'values_lr': 9e-05
	# }
	gaussian_velocity = GaussianSplattingFast(0., 0., 0., 0., np.zeros((1, 2)), dim=2, load_file=os.path.join(cmd_args.dir, f'gaussian_velocity_{cmd_args.start_frame}.pt'))
	N = gaussian_velocity.N
	new_gaussian_velocity = GaussianSplattingFast(0., 0., 0., 0., np.zeros((1, 2)), dim=2, load_file=os.path.join(cmd_args.dir, f'gaussian_velocity_{cmd_args.start_frame}.pt'))
	
	def vorticity_field_2d(x):
		g = gaussian_velocity.gradient(x)
		if len(x.shape) == 1:
			return g[1, 0] - g[0, 1]
		return g[:, 1, 0] - g[:, 0, 1]
	def divergence_field_2d(x):
		g = gaussian_velocity.gradient(x)
		if len(x.shape) == 1:
			return g[0, 0] + g[1, 1]
		return g[:, 0, 0] + g[:, 1, 1]
	
	# near_center = [i for i in range(gaussian_velocity.N) if (gaussian_velocity.positions[i].abs() < torch.tensor([1., .2], device=device)).all()]
	# ellipses_indices = random.sample(list(range(gaussian_velocity.N)), 50) + random.sample(near_center, 5)
	def draw_gaussian_ellipses():
		draw_ellipses(gaussian_velocity)
	
	show_field(gaussian_velocity, x_min_i_gs, x_max_i_gs, y_min_i_gs, y_max_i_gs, dim=2, x_N=30, y_N=30, additional_drawing=draw_gaussian_ellipses, save_filename=os.path.join(cmd_args.dir, f'{cmd_args.start_frame}.png'))
	show_field(original_field(gaussian_velocity), x_min_v, x_max_v, y_min_v, y_max_v, dim=2, x_N=30, y_N=30, save_filename=os.path.join(cmd_args.dir, f'clean_{cmd_args.start_frame}.png'))
	show_field(original_gradient(vorticity_field_2d), x_min_v, x_max_v, y_min_v, y_max_v, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, f'vorticity_{cmd_args.start_frame}.png'))
	show_field(original_gradient(divergence_field_2d), x_min_v, x_max_v, y_min_v, y_max_v, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, f'divergence_{cmd_args.start_frame}.png'))
	
	t, time_step = 0, cmd_args.dt
	cnt = cmd_args.start_frame + 1
	while t < cmd_args.last_time:
		clone_velocity_field(new_gaussian_velocity, gaussian_velocity, default_data_generator, default_test_generator, max_epoch=20000, verbose=1)
		advect_covector_field(new_gaussian_velocity, gaussian_velocity, time_step)
		project(new_gaussian_velocity, AdvectedCovectorField(gaussian_velocity, gaussian_velocity, time_step), default_data_generator, default_test_generator, boundary_generator_1=boundary_sampler[cmd_args.init_cond][0], boundary_generator_2=boundary_sampler[cmd_args.init_cond][1], boundary_lambda=1., max_epoch=20000, verbose=1)
		gaussian_velocity, new_gaussian_velocity = new_gaussian_velocity, gaussian_velocity
		show_field(gaussian_velocity, x_min_i_gs, x_max_i_gs, y_min_i_gs, y_max_i_gs, dim=2, x_N=30, y_N=30, additional_drawing=draw_gaussian_ellipses, save_filename=os.path.join(cmd_args.dir, f'{cnt}.png'))
		show_field(original_field(gaussian_velocity), x_min_v, x_max_v, y_min_v, y_max_v, dim=2, x_N=30, y_N=30, save_filename=os.path.join(cmd_args.dir, f'clean_{cnt}.png'))
		show_field(original_gradient(vorticity_field_2d), x_min_v, x_max_v, y_min_v, y_max_v, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, f'vorticity_{cnt}.png'))
		show_field(original_gradient(divergence_field_2d), x_min_v, x_max_v, y_min_v, y_max_v, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, f'divergence_{cnt}.png'))
		gaussian_velocity.save(os.path.join(cmd_args.dir, f'gaussian_velocity_{cnt}.pt'))
		cnt += 1
		t += time_step
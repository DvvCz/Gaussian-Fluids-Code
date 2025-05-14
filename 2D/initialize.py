import torch
import torch.nn.functional as F
import numpy as np
import time

from init_cond import *
from advance import AdvectedCovectorField


def fit_velocity_with_gradient(gaussian_velocity: GaussianSplattingFast, reference_field, reference_gradient, data_generator, batch_size=512, max_epoch=3000, verbose=1):
	st_time = time.time()
	gaussian_velocity.initialize_optimizers()
	for epoch in range(max_epoch):
		# calculate difference between gaussian_splatting and reference_field
		data = data_generator(batch_size)
		ref_val = reference_field(data)		# referance_field may change data, must execute first
		ref_grad = reference_gradient(data)
		
		val = gaussian_velocity.get_losses(data, ref=ref_val, weight=1.)
		grad = gaussian_velocity.get_grad_losses(data, ref_grad=ref_grad, weight_grad=1.)
		scalings_ratio = torch.exp(-gaussian_velocity.scalings.min(axis=-1).values + gaussian_velocity.scalings.max(axis=-1).values)
		volumes = torch.exp(-gaussian_velocity.scalings.sum(axis=-1))
		loss = F.l1_loss(val, ref_val)
		loss_grad = F.l1_loss(grad, ref_grad)
		loss_aniso = (torch.where(scalings_ratio >= 1.5, scalings_ratio, 1.5) - 1.5).mean()
		loss_vol = ((volumes / volumes.mean() - 1) ** 2).mean()
		(loss_aniso + loss_vol).backward()
		gaussian_velocity.step(loss + loss_grad + loss_aniso + loss_vol)
		
		if verbose and epoch % 100 == 99:
			with torch.no_grad():
				en_time = time.time()
				print(f'loss: {loss}, loss_grad: {loss_grad}, loss_aniso: {loss_aniso}, loss_vol: {loss_vol}, divergence constraint: {((grad[:, 0, 0] + grad[:, 1, 1]) ** 2).sum() / batch_size}')
				print('time:', en_time - st_time)
				st_time = time.time()
	
	if verbose and (max_epoch - 1) % 100 != 99:
		with torch.no_grad():
			en_time = time.time()
			print(f'loss: {loss}, loss_grad: {loss_grad}, loss_aniso: {loss_aniso}, loss_vol: {loss_vol}, divergence constraint: {((grad[:, 0, 0] + grad[:, 1, 1]) ** 2).sum() / batch_size}')
			print('time:', en_time - st_time)


def project(gaussian_velocity: GaussianSplattingFast, reference_field: AdvectedCovectorField, data_generator, test_data_generator, boundary_generator_1=None, boundary_generator_2=None, boundary_lambda=0., batch_size=512, max_epoch=3000, patience=500, verbose=1):
	def gradient_project(g1, g2):
		if (g1 * g2).sum() < 0.:
			n1 = g1 / (g1 ** 2).sum() ** .5
			n2 = g2 / (g2 ** 2).sum() ** .5
			g1 -= (g1 * n2).sum() * n2
			g2 -= (g2 * n1).sum() * n1
	
	positions_org = gaussian_velocity.positions.detach().clone()
	
	def get_losses(data, backward=True):
		weight_vor, weight_div, weight_aniso, weight_vol, weight_delta_pos = 1., 10., 10., 10., .0
		
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
	
	lr_ratio = 1.201956
	gaussian_velocity.set_lr(positions_lr=1e-5 * 10., scalings_lr=1e-5, rotations_lr=1e-5 * lr_ratio, values_lr=1e-5 * 10.)
	gaussian_velocity.initialize_optimizers(patience=50)
	gaussian_velocity.positions_scheduler.factor = .9
	gaussian_velocity.scalings_scheduler.factor = .9
	gaussian_velocity.rotations_scheduler.factor = .9
	gaussian_velocity.values_scheduler.factor = .9
	
	test_data = test_data_generator(gaussian_velocity)
	if verbose:
		_, loss_vor, loss_div, loss_aniso, loss_vol, loos_delta_pos, boundary_constraint = get_losses(test_data, backward=False)
		print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_delta_pos: {loos_delta_pos.item()}, boundary_constraint: {boundary_constraint.item()}')
	
	st_time = time.time()
	check_iter = 100
	min_loss_vor, iter_loss_vor = np.inf, 0
	min_loss_div, iter_loss_div = np.inf, 0
	for epoch in range(max_epoch):
		data = data_generator(batch_size, gaussian_velocity)
		loss_tot, loss_vor, loss_div, loss_aniso, loss_vol, loos_delta_pos, boundary_constraint = get_losses(data)
		gaussian_velocity.step(loss_tot)
		
		if epoch % check_iter == check_iter - 1:
			test_data = test_data_generator(gaussian_velocity)
			_, loss_vor, loss_div, loss_aniso, loss_vol, loos_delta_pos, boundary_constraint = get_losses(test_data, backward=False)
			if verbose:
				print(f'[projection] loss_vor: {loss_vor.mean().item()}, loss_div: {loss_div.mean().item()}, loss_div_max: {loss_div.max().item()}, loss_aniso: {loss_aniso.item()}, loss_vol: {loss_vol.item()}, loss_delta_pos: {loos_delta_pos.item()}, boundary_constraint: {boundary_constraint.item()}, time: {time.time() - st_time}')
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

def init_karman_velocity(gaussian_velocity: GaussianSplattingFast, reference_field, reference_gradient, data_generator, batch_size=512, max_epoch=3000, verbose=1):
	lr_ratio = 1.201956
	gaussian_velocity.set_lr(positions_lr=1.6e-4 * 10., scalings_lr=5e-3, rotations_lr=5e-3 * lr_ratio, values_lr=5e-4 * 10.)
	fit_velocity_with_gradient(gaussian_velocity, reference_field, reference_gradient, data_generator, batch_size, max_epoch, verbose)
	x_min, x_max, y_min, y_max = initialize_domain['karman']
	x_N, y_N = initial_particle_count['karman']
	x_Nvis, y_Nvis = visualize_res[cmd_args.init_cond]
	x_min_gs, x_max_gs, y_min_gs, y_max_gs = x_min * scaling_factor, x_max * scaling_factor, y_min * scaling_factor, y_max * scaling_factor
	tmp_gaussian = GaussianSplattingFast(x_min_gs, x_max_gs, y_min_gs, y_max_gs, get_grid_points(x_min_gs, x_max_gs, y_min_gs, y_max_gs, x_N, y_N).cpu().numpy(), dim=2)
	with torch.no_grad():
		tmp_gaussian.positions = gaussian_velocity.positions.clone()
		tmp_gaussian.scalings = gaussian_velocity.scalings.clone()
		tmp_gaussian.rotations = gaussian_velocity.rotations.clone()
		tmp_gaussian.values = gaussian_velocity.values.clone()
	tmp_gaussian.zero_grad()
	
	def default_data_generator(n, gaussian_splatting: GaussianSplattingFast, restrict=None):
		x_min, x_max, y_min, y_max = advance_domain['karman']
		return (torch.rand_like(gaussian_splatting.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min], device=device) + torch.tensor([x_min, y_min], device=device)) * scaling_factor
	def default_test_generator(gaussian_splatting: GaussianSplattingFast):
		x_min, x_max, y_min, y_max = advance_domain['karman']
		return get_grid_points(x_min, x_max, y_min, y_max, x_Nvis, y_Nvis) * scaling_factor
	
	project(gaussian_velocity, AdvectedCovectorField(tmp_gaussian, tmp_gaussian, 0.), default_data_generator, default_test_generator, boundary_generator_1=boundary_sampler['karman'][0], boundary_generator_2=boundary_sampler['karman'][1], boundary_lambda=10., patience=10000, max_epoch=10000, verbose=1)

def SimulationInitialize():
	x_min, x_max, y_min, y_max = initialize_domain[cmd_args.init_cond]
	x_Nvis, y_Nvis = visualize_res[cmd_args.init_cond]
	x_min_gs, x_max_gs, y_min_gs, y_max_gs = x_min * scaling_factor, x_max * scaling_factor, y_min * scaling_factor, y_max * scaling_factor
	
	ref_velocity = eval(cmd_args.init_cond)
	def velocity_field(x):
		return ref_velocity(x, False)
	def velocity_gradient(x):
		return ref_velocity(x, True)
	def vorticity_field(x):
		g = velocity_gradient(x)
		if len(x.shape) == 1:
			return g[1, 0] - g[0, 1]
		return g[:, 1, 0] - g[:, 0, 1]
	def divergence_field(x):
		g = velocity_gradient(x)
		if len(x.shape) == 1:
			return g[0, 0] + g[1, 1]
		return g[:, 0, 0] + g[:, 1, 1]
	
	show_field(velocity_field, x_min, x_max, y_min, y_max, dim=2, x_N=30, y_N=30, save_filename=os.path.join(cmd_args.dir, 'refvelocity.png'))
	show_field(vorticity_field, x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'refvorticity.png'))
	show_field(divergence_field, x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'refdivergence.png'))
	
	x_N, y_N = initial_particle_count[cmd_args.init_cond]
	gaussian_velocity = GaussianSplattingFast(x_min_gs, x_max_gs, y_min_gs, y_max_gs, get_grid_points(x_min_gs, x_max_gs, y_min_gs, y_max_gs, x_N, y_N).cpu().numpy(), dim=2)
	print(f'Particle count: {gaussian_velocity.N} ({x_N} x {y_N})')
	
	def default_generator(n):
		return (torch.rand_like(gaussian_velocity.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min], device=device) + torch.tensor([x_min, y_min], device=device)) * scaling_factor
	
	def draw_gaussian_ellipses():
		draw_ellipses(gaussian_velocity)
	
	if cmd_args.init_cond == 'karman':
		init_karman_velocity(gaussian_velocity, target_field(velocity_field), target_gradient(velocity_gradient), default_generator, max_epoch=10000)
	else:
		gaussian_velocity.set_lr(positions_lr=1.6e-3, scalings_lr=5e-2, rotations_lr=5e-2, values_lr=5e-3)
		fit_velocity_with_gradient(gaussian_velocity, target_field(velocity_field), target_gradient(velocity_gradient), default_generator, max_epoch=10000)
	gaussian_velocity.save(os.path.join(cmd_args.dir, 'gaussian_velocity_0.pt'))
	show_field(gaussian_velocity, x_min_gs, x_max_gs, y_min_gs, y_max_gs, dim=2, x_N=30, y_N=30, additional_drawing=draw_gaussian_ellipses, save_filename=os.path.join(cmd_args.dir, '0.png'))
	show_field(original_field(gaussian_velocity), x_min, x_max, y_min, y_max, dim=2, x_N=30, y_N=30, save_filename=os.path.join(cmd_args.dir, 'clean_0.png'))
	
	def vorticity_gaussian(x):
		g = gaussian_velocity.gradient(x)
		return g[:, 1, 0] - g[:, 0, 1]
	def divergence_gaussian(x):
		g = gaussian_velocity.gradient(x)
		return g[:, 0, 0] + g[:, 1, 1]
	show_field(original_gradient(vorticity_gaussian), x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'vorticity_0.png'))
	show_field(original_gradient(divergence_gaussian), x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'divergence_0.png'))


if __name__ == '__main__':
	SimulationInitialize()
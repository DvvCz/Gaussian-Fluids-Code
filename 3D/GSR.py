import torch
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import time
import os
import vtk
import vtk.util.numpy_support as vtk_np


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--dir', type=str, default='output_3d')
	parser.add_argument('--start_frame', type=int, default=0)
	parser.add_argument('--boundary', type=float, default=10.)
	parser.add_argument('--init_cond', type=str, default='leapfrog')
	parser.add_argument('--dt', type=float, default=.02)
	parser.add_argument('--last_time', type=float, default=100.)
	return parser.parse_args()
cmd_args = parse_args()
os.makedirs(cmd_args.dir, exist_ok=True)

torch.manual_seed(42)
if cmd_args.device != 'cpu':
	os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.device
	torch.cuda.manual_seed_all(42)
device = torch.device('cpu' if cmd_args.device == 'cpu' else 'cuda')
ti.init(arch=ti.cpu if cmd_args.device == 'cpu' else ti.cuda)

TiArr = ti.types.ndarray()

class GaussianSplatting3D:
	def __init__(self, positions, dim=1, positions_lr=1.6e-3, scalings_lr=5e-2, rotations_lr=5e-2, values_lr=5e-3):
		self.N, self.dim = positions.shape[0], dim
		
		self.positions = torch.tensor(positions, device=device, requires_grad=True)
		self.scalings = torch.zeros((self.N, 3), device=device, requires_grad=True)
		self.rotations = torch.zeros((self.N, 4), device=device)
		self.rotations[:, 0] = 1.
		self.rotations.requires_grad_()
		self.values = torch.zeros((self.N, dim), device=device, requires_grad=True)
		
		self.positions_lr = positions_lr
		self.scalings_lr = scalings_lr
		self.rotations_lr = rotations_lr
		self.values_lr = values_lr
	
	def initialize_optimizers(self, patience=50):
		self.positions_optimizer = torch.optim.Adam([self.positions], lr=self.positions_lr)
		self.positions_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.positions_optimizer, factor=.9, patience=patience)
		self.scalings_optimizer = torch.optim.Adam([self.scalings], lr=self.scalings_lr)
		self.scalings_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.scalings_optimizer, factor=.9, patience=patience)
		self.rotations_optimizer = torch.optim.Adam([self.rotations], lr=self.rotations_lr)
		self.rotations_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.rotations_optimizer, factor=.9, patience=patience)
		self.values_optimizer = torch.optim.Adam([self.values], lr=self.values_lr)
		self.values_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.values_optimizer, factor=.9, patience=patience)
		
		self.optimizers = [
			self.positions_optimizer,
			self.scalings_optimizer,
			self.rotations_optimizer,
			self.values_optimizer,
		]
		self.schedulers = [
			self.positions_scheduler,
			self.scalings_scheduler,
			self.rotations_scheduler,
			self.values_scheduler,
		]
	
	def parameters(self):
		return {
			'positions': self.positions,
			'scalings': self.scalings,
			'rotations': self.rotations,
			'values': self.values
		}
	
	def save(self, filename):
		torch.save(self.parameters(), filename)
	
	def load(self, filename):
		parameters_dict = torch.load(filename, map_location=device)
		self.positions = parameters_dict['positions']
		self.scalings = parameters_dict['scalings']
		self.rotations = parameters_dict['rotations']
		self.values = parameters_dict['values']
		self.N = self.positions.shape[0]
		self.dim = self.values.shape[1]
	
	def get_scaling_matrices(self):
		return torch.diag_embed(torch.exp(self.scalings))
	
	def get_rotation_matrices(self):
		norm = (self.rotations ** 2).sum(axis=-1) ** .5
		q = self.rotations / norm[:, None]
		r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
		R = torch.zeros((self.rotations.shape[0], 3, 3), device=device)
		R[:, 0, 0] = 1 - 2 * (y*y + z*z)
		R[:, 0, 1] = 2 * (x*y - r*z)
		R[:, 0, 2] = 2 * (x*z + r*y)
		R[:, 1, 0] = 2 * (x*y + r*z)
		R[:, 1, 1] = 1 - 2 * (x*x + z*z)
		R[:, 1, 2] = 2 * (y*z - r*x)
		R[:, 2, 0] = 2 * (x*z - r*y)
		R[:, 2, 1] = 2 * (y*z + r*x)
		R[:, 2, 2] = 1 - 2 * (x*x + y*y)
		return R
	
	def get_variances(self):
		S = self.get_scaling_matrices()
		R = self.get_rotation_matrices()
		A = R @ S
		return A @ A.transpose(-1, -2)
	
	def __call__(self, x):
		mu, sigma_inv = self.positions, self.get_variances()
		positions_differences = x[:, None, :] - mu[None, :, :]
		per_splatting_values = self.values * torch.exp(-.5 * positions_differences[:, :, None, :] @ sigma_inv @ positions_differences[:, :, :, None]).squeeze(-1)
		return per_splatting_values.sum(axis=1)
	
	def gradient(self, x, need_val=False):
		mu, sigma_inv = self.positions, self.get_variances()
		positions_differences = x[:, None, :] - mu[None, :, :]
		per_splatting_values = self.values * torch.exp(-.5 * positions_differences[:, :, None, :] @ sigma_inv @ positions_differences[:, :, :, None]).squeeze(-1)
		y = per_splatting_values.sum(axis=1)
		grad = -(per_splatting_values[:, :, :, None] @ (sigma_inv @ positions_differences[:, :, :, None]).transpose(-1, -2)).sum(axis=1)
		return (grad, y) if need_val else grad
	
	def freeze(self):
		self.positions.requires_grad_(False)
		self.scalings.requires_grad_(False)
		self.rotations.requires_grad_(False)
		self.values.requires_grad_(False)
	
	def unfreeze(self):
		self.positions.requires_grad_()
		self.scalings.requires_grad_()
		self.rotations.requires_grad_()
		self.values.requires_grad_()
	
	def zero_grad(self):
		for o in self.optimizers:
			o.zero_grad()
	
	def step(self, metrics):
		for o in self.optimizers:
			o.step()
		for s in self.schedulers:
			s.step(metrics)

@ti.data_oriented
class GaussianSplatting3DFast(GaussianSplatting3D):
	def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, positions, min_grid_scale=None, clamp_threshold=5e-3, dim=1, positions_lr=1e-3, scalings_lr=1e-3, rotations_lr=1e-3, values_lr=1e-3, load_file=None):
		super().__init__(positions, dim, positions_lr, scalings_lr, rotations_lr, values_lr)
		
		if load_file is None:
			self.min_grid_scale = ((x_max - x_min) * (y_max - y_min) * (z_max - z_min) / self.N) ** (1./3.) * 2. if (min_grid_scale is None) else min_grid_scale
			self.clamp_threshold = clamp_threshold
			self.x_min, self.x_max = x_min - self.min_grid_scale, x_max + self.min_grid_scale
			self.y_min, self.y_max = y_min - self.min_grid_scale, y_max + self.min_grid_scale
			self.z_min, self.z_max = z_min - self.min_grid_scale, z_max + self.min_grid_scale
			with torch.no_grad():
				self.scalings += .5 * np.log(-2. * np.log(self.clamp_threshold)) - np.log(self.min_grid_scale)
			self.create_grid_data()
			self.zero_grad()
		else:
			self.load(load_file)
	
	def create_grid_data(self):
		self.grid_size = [int((self.x_max - self.x_min) // self.min_grid_scale) + 1, int((self.y_max - self.y_min) // self.min_grid_scale) + 1, int((self.z_max - self.z_min) // self.min_grid_scale) + 1]
		self.sorted_id = ti.field(dtype=ti.i32, shape=self.N * 5)
		self.grid_offset = ti.field(dtype=ti.i32, shape=self.grid_size)
		self.grid_offset_per_x = ti.field(dtype=ti.i32, shape=self.grid_size[0])
		self.grid_cnt = ti.field(dtype=ti.i32, shape=self.grid_size)
	
	def parameters(self):
		return {
			'positions': self.positions,
			'scalings': self.scalings,
			'rotations': self.rotations,
			'values': self.values,
			'clamp_threshold': self.clamp_threshold,
			'min_grid_scale': self.min_grid_scale,
			'domain_range': (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
		}
	
	def load(self, filename, first_time=True):
		parameters_dict = torch.load(filename, map_location=device)
		self.positions = parameters_dict['positions']
		self.scalings = parameters_dict['scalings']
		self.rotations = parameters_dict['rotations']
		self.values = parameters_dict['values']
		self.N = self.positions.shape[0]
		self.dim = self.values.shape[1]
		self.clamp_threshold = parameters_dict['clamp_threshold']
		self.min_grid_scale = parameters_dict['min_grid_scale']
		self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = parameters_dict['domain_range']
		if first_time:
			self.create_grid_data()
		self.zero_grad()
	
	@ti.kernel
	def reinitialize_grid_ti(self, positions: TiArr, grid_scale: ti.f32):
		for i in range(self.grid_size[0]):
			for j in range(self.grid_size[1]):
				for k in range(self.grid_size[2]):
					self.grid_cnt[i, j, k] = 0
		for i in range(positions.shape[0]):
			if self.x_min <= positions[i, 0] <= self.x_max and self.y_min <= positions[i, 1] <= self.y_max and self.z_min <= positions[i, 2] <= self.z_max:
				idx, idy, idz = int((positions[i, 0] - self.x_min) // grid_scale), int((positions[i, 1] - self.y_min) // grid_scale), int((positions[i, 2] - self.z_min) // grid_scale)
				self.grid_cnt[idx, idy, idz] += 1
		self.grid_offset_per_x[0] = 0
		for i in range(1, self.grid_size[0]):
			self.grid_offset_per_x[i] = 0
			for j in range(self.grid_size[1]):
				for k in range(self.grid_size[2]):
					self.grid_offset_per_x[i] += self.grid_cnt[i-1, j, k]
		for _ in range(1):
			for i in range(1, self.grid_size[0]):
				self.grid_offset_per_x[i] += self.grid_offset_per_x[i-1]
		for i in range(self.grid_size[0]):
			for j in range(self.grid_size[1]):
				for k in range(self.grid_size[2]):
					prev_j, prev_k = j, k
					if prev_k > 0:
						prev_k -= 1
					elif prev_j > 0:
						prev_j -= 1
						prev_k = self.grid_size[2] - 1
					else:
						self.grid_offset[i, j, k] = self.grid_offset_per_x[i]
						continue
					self.grid_offset[i, j, k] = self.grid_offset[i, prev_j, prev_k] + self.grid_cnt[i, prev_j, prev_k]
		for i in range(self.grid_size[0]):
			for j in range(self.grid_size[1]):
				for k in range(self.grid_size[2]):
					self.grid_cnt[i, j, k] = 0
		for i in range(positions.shape[0]):
			if self.x_min <= positions[i, 0] <= self.x_max and self.y_min <= positions[i, 1] <= self.y_max and self.z_min <= positions[i, 2] <= self.z_max:
				idx, idy, idz = int((positions[i, 0] - self.x_min) // grid_scale), int((positions[i, 1] - self.y_min) // grid_scale), int((positions[i, 2] - self.z_min) // grid_scale)
				cur_cnt = ti.atomic_add(self.grid_cnt[idx, idy, idz], 1)
				self.sorted_id[self.grid_offset[idx, idy, idz] + cur_cnt] = i
	
	def reinitialize_grid(self):
		if self.clamp_threshold:
			self.grid_scale = max(np.sqrt(-2. * np.log(self.clamp_threshold)) * np.exp(-self.scalings.min().item()), self.min_grid_scale)
		else:
			self.grid_scale = max(self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)
		self.reinitialize_grid_ti(self.positions, self.grid_scale)
	
	@ti.kernel
	def get_losses_ti(self,
				   positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr,
				   grid_scale: ti.f32,
				   x: TiArr,
				   ref_val: TiArr, weight_val: ti.f32,
				   normals: TiArr, weight_boundary: ti.f32,
				   ref_grad: TiArr, weight_grad: ti.f32,
				   ref_vor: TiArr, weight_vor: ti.f32,
				   ref_hel: TiArr, weight_hel: ti.f32,
				   weight_div: ti.f32,
				   val: TiArr, grad: TiArr,
				   vor_positions_grad: TiArr, vor_scalings_grad: TiArr, vor_rotations_grad: TiArr, vor_values_grad: TiArr,
				   div_positions_grad: TiArr, div_scalings_grad: TiArr, div_rotations_grad: TiArr, div_values_grad: TiArr,
				   stop_gradient: TiArr):
		Q = x.shape[0]
		for j in range(Q):
			idx, idy, idz = int((x[j, 0] - self.x_min) // grid_scale), int((x[j, 1] - self.y_min) // grid_scale), int((x[j, 2] - self.z_min) // grid_scale)
			for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
				for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
					for gk in range(max(idz - 1, 0), min(idz + 1, self.grid_size[2] - 1) + 1):
						for i_id in range(self.grid_offset[gi, gj, gk], self.grid_offset[gi, gj, gk] + self.grid_cnt[gi, gj, gk]):
							i = self.sorted_id[i_id]
							delta_x = tm.vec3([x[j, 0], x[j, 1], x[j, 2]]) - tm.vec3([positions[i, 0], positions[i, 1], positions[i, 2]])
							q = tm.normalize(tm.vec4(rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
							R = tm.mat3([
								[1. - 2. * (q[2] * q[2] + q[3] * q[3]), 2. * (q[1] * q[2] - q[0] * q[3]), 2. * (q[1] * q[3] + q[0] * q[2])],
								[2. * (q[1] * q[2] + q[0] * q[3]), 1. - 2. * (q[1] * q[1] + q[3] * q[3]), 2. * (q[2] * q[3] - q[0] * q[1])],
								[2. * (q[1] * q[3] - q[0] * q[2]), 2. * (q[2] * q[3] + q[0] * q[1]), 1. - 2. * (q[1] * q[1] + q[2] * q[2])]
							])
							S2 = tm.mat3([
								[tm.exp(2. * scalings[i, 0]), 0., 0.],
								[0., tm.exp(2. * scalings[i, 1]), 0.],
								[0., 0., tm.exp(2. * scalings[i, 2])]
							])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_x @ cov_inv @ delta_x)
							if gaussian >= self.clamp_threshold:
								grad_gaussian = -gaussian * (cov_inv @ delta_x)
								for d in range(self.dim):
									val[j, d] += values[i, d] * (gaussian - self.clamp_threshold)
									if grad.shape[0] > 0:
										grad[j, d, 0] += values[i, d] * grad_gaussian[0]
										grad[j, d, 1] += values[i, d] * grad_gaussian[1]
										grad[j, d, 2] += values[i, d] * grad_gaussian[2]
		if weight_val == 0. and weight_boundary == 0. and weight_grad == 0. and weight_vor == 0. and weight_div == 0.:
			Q = 0
		for j in range(Q):
			idx, idy, idz = int((x[j, 0] - self.x_min) // grid_scale), int((x[j, 1] - self.y_min) // grid_scale), int((x[j, 2] - self.z_min) // grid_scale)
			for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
				for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
					for gk in range(max(idz - 1, 0), min(idz + 1, self.grid_size[2] - 1) + 1):
						for i_id in range(self.grid_offset[gi, gj, gk], self.grid_offset[gi, gj, gk] + self.grid_cnt[gi, gj, gk]):
							i = self.sorted_id[i_id]
							if stop_gradient[i]:
								continue
							delta_x = tm.vec3([x[j, 0], x[j, 1], x[j, 2]]) - tm.vec3([positions[i, 0], positions[i, 1], positions[i, 2]])
							r = tm.vec4(rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3])
							q = tm.normalize(r)
							R = tm.mat3([
								[1. - 2. * (q[2] * q[2] + q[3] * q[3]), 2. * (q[1] * q[2] - q[0] * q[3]), 2. * (q[1] * q[3] + q[0] * q[2])],
								[2. * (q[1] * q[2] + q[0] * q[3]), 1. - 2. * (q[1] * q[1] + q[3] * q[3]), 2. * (q[2] * q[3] - q[0] * q[1])],
								[2. * (q[1] * q[3] - q[0] * q[2]), 2. * (q[2] * q[3] + q[0] * q[1]), 1. - 2. * (q[1] * q[1] + q[2] * q[2])]
							])
							S2 = tm.mat3([
								[tm.exp(2. * scalings[i, 0]), 0., 0.],
								[0., tm.exp(2. * scalings[i, 1]), 0.],
								[0., 0., tm.exp(2. * scalings[i, 2])]
							])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_x @ cov_inv @ delta_x)
							if gaussian >= self.clamp_threshold:
								# Prep args
								grad_gaussian = -gaussian * (cov_inv @ delta_x)
								d_R_q0 = tm.mat3([
									[0., -2. * q[3], 2. * q[2]],
									[2. * q[3], 0., -2. * q[1]],
									[-2. * q[2], 2. * q[1], 0.]
								])
								d_R_q1 = tm.mat3([
									[0., 2. * q[2], 2. * q[3]],
									[2. * q[2], -4. * q[1], -2. * q[0]],
									[2. * q[3], 2. * q[0], -4. * q[1]]
								])
								d_R_q2 = tm.mat3([
									[-4. * q[2], 2. * q[1], 2. * q[0]],
									[2. * q[1], 0., 2. * q[3]],
									[-2. * q[0], 2. * q[3], -4. * q[2]]
								])
								d_R_q3 = tm.mat3([
									[-4. * q[3], -2. * q[0], 2. * q[1]],
									[2. * q[0], -4. * q[3], 2. * q[2]],
									[2. * q[1], 2. * q[2], 0.]
								])
								r_length = tm.length(r)
								d_R_r0 = -r[0] / r_length ** 3 * (r[0] * d_R_q0 + r[1] * d_R_q1 + r[2] * d_R_q2 + r[3] * d_R_q3) + d_R_q0 / r_length
								d_R_r1 = -r[1] / r_length ** 3 * (r[0] * d_R_q0 + r[1] * d_R_q1 + r[2] * d_R_q2 + r[3] * d_R_q3) + d_R_q1 / r_length
								d_R_r2 = -r[2] / r_length ** 3 * (r[0] * d_R_q0 + r[1] * d_R_q1 + r[2] * d_R_q2 + r[3] * d_R_q3) + d_R_q2 / r_length
								d_R_r3 = -r[3] / r_length ** 3 * (r[0] * d_R_q0 + r[1] * d_R_q1 + r[2] * d_R_q2 + r[3] * d_R_q3) + d_R_q3 / r_length
								d_cov_inv_r0 = tm.mat3(0.)
								d_cov_inv_r1 = tm.mat3(0.)
								d_cov_inv_r2 = tm.mat3(0.)
								d_cov_inv_r3 = tm.mat3(0.)
								RS2 = R @ S2
								for kk in range(3):
									for ll in range(3):
										for ii in range(3):
											d_cov_inv_r0[kk, ll] += RS2[kk, ii] * d_R_r0[ll, ii] + RS2[ll, ii] * d_R_r0[kk, ii]
											d_cov_inv_r1[kk, ll] += RS2[kk, ii] * d_R_r1[ll, ii] + RS2[ll, ii] * d_R_r1[kk, ii]
											d_cov_inv_r2[kk, ll] += RS2[kk, ii] * d_R_r2[ll, ii] + RS2[ll, ii] * d_R_r2[kk, ii]
											d_cov_inv_r3[kk, ll] += RS2[kk, ii] * d_R_r3[ll, ii] + RS2[ll, ii] * d_R_r3[kk, ii]
								R_col0, R_col1, R_col2 = tm.vec3([R[0, 0], R[1, 0], R[2, 0]]), tm.vec3([R[0, 1], R[1, 1], R[2, 1]]), tm.vec3([R[0, 2], R[1, 2], R[2, 2]])
								d_cov_inv_s0 = 2. * tm.exp(2. * scalings[i, 0]) * R_col0.outer_product(R_col0)
								d_cov_inv_s1 = 2. * tm.exp(2. * scalings[i, 1]) * R_col1.outer_product(R_col1)
								d_cov_inv_s2 = 2. * tm.exp(2. * scalings[i, 2]) * R_col2.outer_product(R_col2)
								d_gaussian_r0 = -.5 * gaussian * (delta_x @ d_cov_inv_r0 @ delta_x)
								d_gaussian_r1 = -.5 * gaussian * (delta_x @ d_cov_inv_r1 @ delta_x)
								d_gaussian_r2 = -.5 * gaussian * (delta_x @ d_cov_inv_r2 @ delta_x)
								d_gaussian_r3 = -.5 * gaussian * (delta_x @ d_cov_inv_r3 @ delta_x)
								d_gaussian_s0 = -.5 * gaussian * (delta_x @ d_cov_inv_s0 @ delta_x)
								d_gaussian_s1 = -.5 * gaussian * (delta_x @ d_cov_inv_s1 @ delta_x)
								d_gaussian_s2 = -.5 * gaussian * (delta_x @ d_cov_inv_s2 @ delta_x)
								cov_inv_times_delta_x = cov_inv @ delta_x
								d_grad_gaussian_position = tm.mat3(0.)
								d_grad_gaussian_s0 = tm.vec3(0.)
								d_grad_gaussian_s1 = tm.vec3(0.)
								d_grad_gaussian_s2 = tm.vec3(0.)
								d_grad_gaussian_r0 = tm.vec3(0.)
								d_grad_gaussian_r1 = tm.vec3(0.)
								d_grad_gaussian_r2 = tm.vec3(0.)
								d_grad_gaussian_r3 = tm.vec3(0.)
								if weight_grad != 0. or weight_vor != 0. or weight_div != 0.:
									d_grad_gaussian_position = gaussian * (cov_inv @ (tm.eye(3) - delta_x.outer_product(delta_x) @ cov_inv))
									d_grad_gaussian_s0 = -d_gaussian_s0 * cov_inv_times_delta_x - gaussian * (d_cov_inv_s0 @ delta_x)
									d_grad_gaussian_s1 = -d_gaussian_s1 * cov_inv_times_delta_x - gaussian * (d_cov_inv_s1 @ delta_x)
									d_grad_gaussian_s2 = -d_gaussian_s2 * cov_inv_times_delta_x - gaussian * (d_cov_inv_s2 @ delta_x)
									d_grad_gaussian_r0 = -d_gaussian_r0 * cov_inv_times_delta_x - gaussian * (d_cov_inv_r0 @ delta_x)
									d_grad_gaussian_r1 = -d_gaussian_r1 * cov_inv_times_delta_x - gaussian * (d_cov_inv_r1 @ delta_x)
									d_grad_gaussian_r2 = -d_gaussian_r2 * cov_inv_times_delta_x - gaussian * (d_cov_inv_r2 @ delta_x)
									d_grad_gaussian_r3 = -d_gaussian_r3 * cov_inv_times_delta_x - gaussian * (d_cov_inv_r3 @ delta_x)
								
								# Gradients of value loss
								if weight_val != 0.:
									w = weight_val / (self.dim * Q)
									value_dot_sign = 0.
									for d in range(self.dim):
										value_dot_sign += values[i, d] * tm.sign(val[j, d] - ref_val[j, d])
										values.grad[i, d] += w * (gaussian - self.clamp_threshold) * tm.sign(val[j, d] - ref_val[j, d])
									positions.grad[i, 0] += w * gaussian * value_dot_sign * cov_inv_times_delta_x[0]
									positions.grad[i, 1] += w * gaussian * value_dot_sign * cov_inv_times_delta_x[1]
									positions.grad[i, 2] += w * gaussian * value_dot_sign * cov_inv_times_delta_x[2]
									scalings.grad[i, 0] += w * value_dot_sign * d_gaussian_s0
									scalings.grad[i, 1] += w * value_dot_sign * d_gaussian_s1
									scalings.grad[i, 2] += w * value_dot_sign * d_gaussian_s2
									rotations.grad[i, 0] += w * value_dot_sign * d_gaussian_r0
									rotations.grad[i, 1] += w * value_dot_sign * d_gaussian_r1
									rotations.grad[i, 2] += w * value_dot_sign * d_gaussian_r2
									rotations.grad[i, 3] += w * value_dot_sign * d_gaussian_r3
								
								# Gradients of boundary constraint
								if weight_boundary != 0.:
									w = weight_boundary / Q
									sign_val_dot_normal = 0.
									value_dot_normal = 0.
									for d in range(self.dim):
										sign_val_dot_normal += val[j, d] * normals[j, d]
										value_dot_normal += values[i, d] * normals[j, d]
									sign_val_dot_normal = tm.sign(sign_val_dot_normal)
									for d in range(self.dim):
										values.grad[i, d] += w * sign_val_dot_normal * (gaussian - self.clamp_threshold) * normals[j, d]
									positions.grad[i, 0] += w * sign_val_dot_normal * value_dot_normal * gaussian * cov_inv_times_delta_x[0]
									positions.grad[i, 1] += w * sign_val_dot_normal * value_dot_normal * gaussian * cov_inv_times_delta_x[1]
									positions.grad[i, 2] += w * sign_val_dot_normal * value_dot_normal * gaussian * cov_inv_times_delta_x[2]
									scalings.grad[i, 0] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_s0
									scalings.grad[i, 1] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_s1
									scalings.grad[i, 2] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_s2
									rotations.grad[i, 0] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_r0
									rotations.grad[i, 1] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_r1
									rotations.grad[i, 2] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_r2
									rotations.grad[i, 3] += w * sign_val_dot_normal * value_dot_normal * d_gaussian_r3
								
								# Gradients of gradient loss
								if weight_grad != 0.:
									w = weight_grad / (3 * self.dim * Q)
									value_times_sign = tm.vec3(0.)
									for d in range(self.dim):
										value_times_sign += values[i, d] * tm.sign(tm.vec3([grad[j, d, 0], grad[j, d, 1], grad[j, d, 2]]) - tm.vec3([ref_grad[j, d, 0], ref_grad[j, d, 1], ref_grad[j, d, 2]]))
										values.grad[i, d] += w * (tm.sign(tm.vec3([grad[j, d, 0], grad[j, d, 1], grad[j, d, 2]]) - tm.vec3([ref_grad[j, d, 0], ref_grad[j, d, 1], ref_grad[j, d, 2]])) @ grad_gaussian)
									positions.grad[i, 0] += w * (value_times_sign @ tm.vec3([d_grad_gaussian_position[0, 0], d_grad_gaussian_position[1, 0], d_grad_gaussian_position[2, 0]]))
									positions.grad[i, 1] += w * (value_times_sign @ tm.vec3([d_grad_gaussian_position[0, 1], d_grad_gaussian_position[1, 1], d_grad_gaussian_position[2, 1]]))
									positions.grad[i, 2] += w * (value_times_sign @ tm.vec3([d_grad_gaussian_position[0, 2], d_grad_gaussian_position[1, 2], d_grad_gaussian_position[2, 2]]))
									scalings.grad[i, 0] += w * (value_times_sign @ d_grad_gaussian_s0)
									scalings.grad[i, 1] += w * (value_times_sign @ d_grad_gaussian_s1)
									scalings.grad[i, 2] += w * (value_times_sign @ d_grad_gaussian_s2)
									rotations.grad[i, 0] += w * (value_times_sign @ d_grad_gaussian_r0)
									rotations.grad[i, 1] += w * (value_times_sign @ d_grad_gaussian_r1)
									rotations.grad[i, 2] += w * (value_times_sign @ d_grad_gaussian_r2)
									rotations.grad[i, 3] += w * (value_times_sign @ d_grad_gaussian_r3)
								
								# Gradient of vorticity and helicity loss
								if weight_vor + weight_hel != 0. and self.dim == 3:
									# Gradient of vorticity loss
									w = weight_vor / (3 * Q)
									vor = tm.vec3([grad[j, 2, 1] - grad[j, 1, 2], grad[j, 0, 2] - grad[j, 2, 0], grad[j, 1, 0] - grad[j, 0, 1]])
									d_vor0_grad = tm.mat3([
										[0., 0., 0.],
										[0., 0., -1.],
										[0., 1., 0.]
									])
									d_vor1_grad = tm.mat3([
										[0., 0., 1.],
										[0., 0., 0.],
										[-1., 0., 0.]
									])
									d_vor2_grad = tm.mat3([
										[0., -1., 0.],
										[1., 0., 0.],
										[0., 0., 0.]
									])
									sign_vor_diff = tm.sign(vor - tm.vec3([ref_vor[j, 0], ref_vor[j, 1], ref_vor[j, 2]]))
									M_vor = sign_vor_diff[0] * d_vor0_grad + sign_vor_diff[1] * d_vor1_grad + sign_vor_diff[2] * d_vor2_grad
									value_times_M_vor = tm.vec3([values[i, 0], values[i, 1], values[i, 2]]) @ M_vor
									vor_values_grad[i, 0] += w * (tm.vec3([M_vor[0, 0], M_vor[0, 1], M_vor[0, 2]]) @ grad_gaussian)
									vor_values_grad[i, 1] += w * (tm.vec3([M_vor[1, 0], M_vor[1, 1], M_vor[1, 2]]) @ grad_gaussian)
									vor_values_grad[i, 2] += w * (tm.vec3([M_vor[2, 0], M_vor[2, 1], M_vor[2, 2]]) @ grad_gaussian)
									vor_positions_grad[i, 0] += w * (value_times_M_vor @ tm.vec3([d_grad_gaussian_position[0, 0], d_grad_gaussian_position[1, 0], d_grad_gaussian_position[2, 0]]))
									vor_positions_grad[i, 1] += w * (value_times_M_vor @ tm.vec3([d_grad_gaussian_position[0, 1], d_grad_gaussian_position[1, 1], d_grad_gaussian_position[2, 1]]))
									vor_positions_grad[i, 2] += w * (value_times_M_vor @ tm.vec3([d_grad_gaussian_position[0, 2], d_grad_gaussian_position[1, 2], d_grad_gaussian_position[2, 2]]))
									vor_scalings_grad[i, 0] += w * (value_times_M_vor @ d_grad_gaussian_s0)
									vor_scalings_grad[i, 1] += w * (value_times_M_vor @ d_grad_gaussian_s1)
									vor_scalings_grad[i, 2] += w * (value_times_M_vor @ d_grad_gaussian_s2)
									vor_rotations_grad[i, 0] += w * (value_times_M_vor @ d_grad_gaussian_r0)
									vor_rotations_grad[i, 1] += w * (value_times_M_vor @ d_grad_gaussian_r1)
									vor_rotations_grad[i, 2] += w * (value_times_M_vor @ d_grad_gaussian_r2)
									vor_rotations_grad[i, 3] += w * (value_times_M_vor @ d_grad_gaussian_r3)
									# Gradient of helicity loss
									w = weight_hel / Q
									sign_hel_diff = tm.sign(tm.vec3([val[j, 0], val[j, 1], val[j, 2]]) @ vor - ref_hel[j])
									M_hel = val[j, 0] * d_vor0_grad + val[j, 1] * d_vor1_grad + val[j, 2] * d_vor2_grad
									value_times_M_hel = tm.vec3([values[i, 0], values[i, 1], values[i, 2]]) @ M_hel
									d_hel_value0 = (gaussian - self.clamp_threshold) * (grad[j, 2, 1] - grad[j, 1, 2]) + tm.vec3([M_hel[0, 0], M_hel[0, 1], M_hel[0, 2]]) @ grad_gaussian
									d_hel_value1 = (gaussian - self.clamp_threshold) * (grad[j, 0, 2] - grad[j, 2, 0]) + tm.vec3([M_hel[1, 0], M_hel[1, 1], M_hel[1, 2]]) @ grad_gaussian
									d_hel_value2 = (gaussian - self.clamp_threshold) * (grad[j, 1, 0] - grad[j, 0, 1]) + tm.vec3([M_hel[2, 0], M_hel[2, 1], M_hel[2, 2]]) @ grad_gaussian
									value_dot_vor = tm.vec3([values[i, 0], values[i, 1], values[i, 2]]) @ vor
									d_hel_position0 = gaussian * cov_inv_times_delta_x[0] * value_dot_vor + value_times_M_hel @ tm.vec3([d_grad_gaussian_position[0, 0], d_grad_gaussian_position[0, 1], d_grad_gaussian_position[0, 2]])
									d_hel_position1 = gaussian * cov_inv_times_delta_x[1] * value_dot_vor + value_times_M_hel @ tm.vec3([d_grad_gaussian_position[1, 0], d_grad_gaussian_position[1, 1], d_grad_gaussian_position[1, 2]])
									d_hel_position2 = gaussian * cov_inv_times_delta_x[2] * value_dot_vor + value_times_M_hel @ tm.vec3([d_grad_gaussian_position[2, 0], d_grad_gaussian_position[2, 1], d_grad_gaussian_position[2, 2]])
									d_hel_s0 = d_gaussian_s0 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_s0
									d_hel_s1 = d_gaussian_s1 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_s1
									d_hel_s2 = d_gaussian_s2 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_s2
									d_hel_r0 = d_gaussian_r0 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_r0
									d_hel_r1 = d_gaussian_r1 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_r1
									d_hel_r2 = d_gaussian_r2 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_r2
									d_hel_r3 = d_gaussian_r3 * value_dot_vor + value_times_M_hel @ d_grad_gaussian_r3
									vor_values_grad[i, 0] += w * sign_hel_diff * d_hel_value0
									vor_values_grad[i, 1] += w * sign_hel_diff * d_hel_value1
									vor_values_grad[i, 2] += w * sign_hel_diff * d_hel_value2
									vor_positions_grad[i, 0] += w * sign_hel_diff * d_hel_position0
									vor_positions_grad[i, 1] += w * sign_hel_diff * d_hel_position1
									vor_positions_grad[i, 2] += w * sign_hel_diff * d_hel_position2
									vor_scalings_grad[i, 0] += w * sign_hel_diff * d_hel_s0
									vor_scalings_grad[i, 1] += w * sign_hel_diff * d_hel_s1
									vor_scalings_grad[i, 2] += w * sign_hel_diff * d_hel_s2
									vor_rotations_grad[i, 0] += w * sign_hel_diff * d_hel_r0
									vor_rotations_grad[i, 1] += w * sign_hel_diff * d_hel_r1
									vor_rotations_grad[i, 2] += w * sign_hel_diff * d_hel_r2
									vor_rotations_grad[i, 3] += w * sign_hel_diff * d_hel_r3
								
								# Gradient of divergence loss
								if weight_div != 0. and self.dim == 3:
									w = weight_div / Q
									# M_div = tm.sign(grad[j, 0, 0] + grad[j, 1, 1] + grad[j, 2, 2])
									M_div = 2. * (grad[j, 0, 0] + grad[j, 1, 1] + grad[j, 2, 2])
									value = tm.vec3([values[i, 0], values[i, 1], values[i, 2]])
									div_values_grad[i, 0] += w * M_div * grad_gaussian[0]
									div_values_grad[i, 1] += w * M_div * grad_gaussian[1]
									div_values_grad[i, 2] += w * M_div * grad_gaussian[2]
									div_positions_grad[i, 0] += w * M_div * (value @ tm.vec3([d_grad_gaussian_position[0, 0], d_grad_gaussian_position[1, 0], d_grad_gaussian_position[2, 0]]))
									div_positions_grad[i, 1] += w * M_div * (value @ tm.vec3([d_grad_gaussian_position[0, 1], d_grad_gaussian_position[1, 1], d_grad_gaussian_position[2, 1]]))
									div_positions_grad[i, 2] += w * M_div * (value @ tm.vec3([d_grad_gaussian_position[0, 2], d_grad_gaussian_position[1, 2], d_grad_gaussian_position[2, 2]]))
									div_scalings_grad[i, 0] += w * M_div * (value @ d_grad_gaussian_s0)
									div_scalings_grad[i, 1] += w * M_div * (value @ d_grad_gaussian_s1)
									div_scalings_grad[i, 2] += w * M_div * (value @ d_grad_gaussian_s2)
									div_rotations_grad[i, 0] += w * M_div * (value @ d_grad_gaussian_r0)
									div_rotations_grad[i, 1] += w * M_div * (value @ d_grad_gaussian_r1)
									div_rotations_grad[i, 2] += w * M_div * (value @ d_grad_gaussian_r2)
									div_rotations_grad[i, 3] += w * M_div * (value @ d_grad_gaussian_r3)
	
	def get_losses(self, x, discard_grad=False,
				ref_val=None, weight_val=0.,
				normals=None, weight_boundary=0.,
				ref_grad=None, weight_grad=0.,
				ref_vor=None, weight_vor=0.,
				ref_hel=None, weight_hel=0.,
				weight_div=0.,
				vor_positions_grad=None, vor_scalings_grad=None, vor_rotations_grad=None, vor_values_grad=None,
				div_positions_grad=None, div_scalings_grad=None, div_rotations_grad=None, div_values_grad=None,
				stop_gradient=None):
		val = torch.zeros((x.shape[0], self.dim), device=device)
		grad = torch.zeros((0 if discard_grad else x.shape[0], self.dim, 3), device=device)
		if weight_val == 0.:
			ref_val = torch.zeros_like(val, device=device)
		if weight_boundary == 0.:
			normals = torch.zeros_like(val, device=device)
		if weight_grad == 0.:
			ref_grad = torch.zeros_like(grad, device=device)
		if weight_vor == 0.:
			ref_vor = torch.zeros((x.shape[0], 3), device=device)
		if weight_hel == 0.:
			ref_hel = torch.zeros(x.shape[0], device=device)
		if vor_positions_grad is None:
			vor_positions_grad = self.positions.grad
		if vor_scalings_grad is None:
			vor_scalings_grad = self.scalings.grad
		if vor_rotations_grad is None:
			vor_rotations_grad = self.rotations.grad
		if vor_values_grad is None:
			vor_values_grad = self.values.grad
		if div_positions_grad is None:
			div_positions_grad = self.positions.grad
		if div_scalings_grad is None:
			div_scalings_grad = self.scalings.grad
		if div_rotations_grad is None:
			div_rotations_grad = self.rotations.grad
		if div_values_grad is None:
			div_values_grad = self.values.grad
		if stop_gradient is None:
			stop_gradient = torch.zeros((self.positions.shape[0],), dtype=torch.int32, device=device)
		self.get_losses_ti(
			self.positions, self.scalings, self.rotations, self.values,
			self.grid_scale,
			x,
			ref_val, weight_val,
			normals, weight_boundary,
			ref_grad, weight_grad,
			ref_vor, weight_vor,
			ref_hel, weight_hel,
			weight_div,
			val, grad,
			vor_positions_grad, vor_scalings_grad, vor_rotations_grad, vor_values_grad,
			div_positions_grad, div_scalings_grad, div_rotations_grad, div_values_grad,
			stop_gradient
		)
		return val if discard_grad else (val, grad)
	
	@ti.func
	def get_3d_val_grad_ti(self,
					 positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr, grid_scale: ti.f32,
					 x: tm.vec3):
		val = tm.vec3(0.)
		grad = tm.mat3(0.)
		idx, idy, idz = int((x[0] - self.x_min) // grid_scale), int((x[1] - self.y_min) // grid_scale), int((x[2] - self.z_min) // grid_scale)
		for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
			for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
				for gk in range(max(idz - 1, 0), min(idz + 1, self.grid_size[2] - 1) + 1):
					for i_id in range(self.grid_offset[gi, gj, gk], self.grid_offset[gi, gj, gk] + self.grid_cnt[gi, gj, gk]):
						i = self.sorted_id[i_id]
						delta_x = tm.vec3([x[0], x[1], x[2]]) - tm.vec3([positions[i, 0], positions[i, 1], positions[i, 2]])
						q = tm.normalize(tm.vec4(rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
						R = tm.mat3([
							[1. - 2. * (q[2] * q[2] + q[3] * q[3]), 2. * (q[1] * q[2] - q[0] * q[3]), 2. * (q[1] * q[3] + q[0] * q[2])],
							[2. * (q[1] * q[2] + q[0] * q[3]), 1. - 2. * (q[1] * q[1] + q[3] * q[3]), 2. * (q[2] * q[3] - q[0] * q[1])],
							[2. * (q[1] * q[3] - q[0] * q[2]), 2. * (q[2] * q[3] + q[0] * q[1]), 1. - 2. * (q[1] * q[1] + q[2] * q[2])]
						])
						S2 = tm.mat3([
							[tm.exp(2. * scalings[i, 0]), 0., 0.],
							[0., tm.exp(2. * scalings[i, 1]), 0.],
							[0., 0., tm.exp(2. * scalings[i, 2])]
						])
						cov_inv = R @ S2 @ R.transpose()
						gaussian = tm.exp(-.5 * delta_x @ cov_inv @ delta_x)
						if gaussian >= self.clamp_threshold:
							grad_gaussian = -gaussian * (cov_inv @ delta_x)
							for d in range(self.dim):
								val[d] += values[i, d] * (gaussian - self.clamp_threshold)
								grad[d, 0] += values[i, d] * grad_gaussian[0]
								grad[d, 1] += values[i, d] * grad_gaussian[1]
								grad[d, 2] += values[i, d] * grad_gaussian[2]
		return val, grad
	
	@ti.kernel
	def advection_rk4_ti(self,
					  positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr, grid_scale: ti.f32,
					  start_pos: TiArr, dt: ti.f32,
					  goal_pos: TiArr, deformation: TiArr, goal_val: TiArr, goal_grad: TiArr):
		for j in range(start_pos.shape[0]):
			x = tm.vec3([start_pos[j, 0], start_pos[j, 1], start_pos[j, 2]])
			v, dv = self.get_3d_val_grad_ti(positions, scalings, rotations, values, grid_scale, x)
			phi1 = x + dt * .5 * v
			v1, dv1 = self.get_3d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi1)
			phi2 = x + dt * .5 * v1
			v2, dv2 = self.get_3d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi2)
			phi3 = x + dt * v2
			v3, dv3 = self.get_3d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi3)
			phi = x + dt / 6. * (v + 2. * v1 + 2. * v2 + v3)
			goal_pos[j, 0], goal_pos[j, 1], goal_pos[j, 2] = phi[0], phi[1], phi[2]
			if deformation.shape[0] > 0:
				dphi1 = tm.eye(3) + dt * .5 * dv
				dv1_x_dphi1 = dv1 @ dphi1
				dphi2 = tm.eye(3) + dt * .5 * dv1_x_dphi1
				dv2_x_dphi2 = dv2 @ dphi2
				dphi3 = tm.eye(3) + dt * dv2_x_dphi2
				dphi = tm.eye(3) + dt / 6. * (dv + 2. * dv1_x_dphi1 + 2. * dv2_x_dphi2 + dv3 @ dphi3)
				for k in range(3):
					for l in range(3):
						deformation[j, k, l] = dphi[k, l]
			if goal_val.shape[0] > 0 and goal_grad.shape[0] > 0:
				v_phi, dv_phi = self.get_3d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi)
				goal_val[j, 0], goal_val[j, 1], goal_val[j, 2] = v_phi[0], v_phi[1], v_phi[2]
				for k in range(3):
					for l in range(3):
						goal_grad[j, k, l] = dv_phi[k, l]
	
	def advection_rk4(self, start_pos, dt, pos_only=True):
		goal_pos = torch.zeros_like(start_pos, device=device)
		deformation = torch.zeros((0 if pos_only else start_pos.shape[0], 3, 3), device=device)
		goal_val = torch.zeros((0 if pos_only else start_pos.shape[0], 3), device=device)
		goal_grad = torch.zeros((0 if pos_only else start_pos.shape[0], 3, 3), device=device)
		self.advection_rk4_ti(
			self.positions, self.scalings, self.rotations, self.values, self.grid_scale,
			start_pos, dt,
			goal_pos, deformation, goal_val, goal_grad
		)
		return goal_pos if pos_only else (goal_pos, deformation, goal_val, goal_grad)
	
	@ti.kernel
	def get_all_neighbors_ti(self, x: TiArr, positions: TiArr, grid_scale: ti.f32, mark: TiArr):
		for j in range(x.shape[0]):
			idx, idy, idz = int((x[j, 0] - self.x_min) // grid_scale), int((x[j, 1] - self.y_min) // grid_scale), int((x[j, 2] - self.z_min) // grid_scale)
			for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
				for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
					for gk in range(max(idz - 1, 0), min(idz + 1, self.grid_size[2] - 1) + 1):
						for i_id in range(self.grid_offset[gi, gj, gk], self.grid_offset[gi, gj, gk] + self.grid_cnt[gi, gj, gk]):
							i = self.sorted_id[i_id]
							delta_x = tm.vec3([x[j, 0], x[j, 1], x[j, 2]]) - tm.vec3([positions[i, 0], positions[i, 1], positions[i, 2]])
							if tm.length(delta_x) <= grid_scale:
								mark[i] = 1
	
	def get_all_neighbors(self, x):
		mark = torch.zeros((self.positions.shape[0],), dtype=torch.int32, device=device)
		self.get_all_neighbors_ti(x, self.positions, self.grid_scale, mark)
		return mark
	
	def __call__(self, x):
		return self.get_losses(x, discard_grad=True)
	
	def gradient(self, x, need_val=False):
		val, grad = self.get_losses(x)
		return (grad, val) if need_val else grad
	
	def zero_grad(self):
		if not hasattr(self, 'optimizers'):
			self.initialize_optimizers()
		for param in super().parameters().values():
			if param.grad is None:
				param.grad = torch.zeros_like(param, device=device)
			else:
				param.grad.zero_()
		self.reinitialize_grid()
	
	def step(self, metrics):
		super().step(metrics)
		self.zero_grad()


def get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_N, y_N, z_N):
	X = torch.linspace(x_min, x_max, x_N, device=device)
	Y = torch.linspace(y_min, y_max, y_N, device=device)
	Z = torch.linspace(z_min, z_max, z_N, device=device)
	X_, Y_, Z_ = torch.meshgrid(X, Y, Z, indexing='ij')
	XYZ = torch.stack((X_, Y_, Z_), dim=-1).reshape(-1, 3)
	return XYZ.contiguous()


def write_vti(field, x_min, x_max, y_min, y_max, z_min, z_max, save_filename: str, x_N=30, y_N=30, z_N=30):
	XYZ = get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_N, y_N, z_N)
	V = field(XYZ).reshape(x_N, y_N, z_N)
	
	V_np = V.detach().cpu().numpy()
	vtk_data = vtk.vtkImageData()
	vtk_data.SetDimensions(V_np.shape)
	vtk_data.SetOrigin(x_min, y_min, z_min)
	vtk_data.SetSpacing((x_max - x_min) / x_N, (y_max - y_min) / y_N, (z_max - z_min) / z_N)
	vtk_data.GetPointData().SetScalars(vtk_np.numpy_to_vtk(V_np.ravel(order='F'), deep=True))
	
	writer = vtk.vtkXMLImageDataWriter()
	writer.SetFileName(save_filename)
	writer.SetInputData(vtk_data)
	writer.Write()


def write_obj(gs: GaussianSplatting3D, save_filename: str):
	with open(save_filename, 'w') as fd:
		for i in range(gs.positions.shape[0]):
			fd.write(f'v {gs.positions[i, 0]} {gs.positions[i, 1]} {gs.positions[i, 2]}\n')
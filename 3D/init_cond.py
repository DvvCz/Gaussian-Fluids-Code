import torch

from GSR import *
from mesh_sampler import MeshSampler


'''
Domain
'''

# [x_min, x_max, y_min, y_max, z_min, z_max]

domain = {
	'leapfrog': (0., 1., 0., 1., 0., 1.),
	'single_vortex_ring': (0., 1., 0., 1., 0., 1.),
	'ring_collide': (0., 1., 0., 1., 0., 1.),
	'ring_with_obstacle': (0., 1., 0., 1., 0., 1.)
}

initial_particle_count = {
	'leapfrog': (10, 10, 10),
	'single_vortex_ring': (40, 40, 40),
	'ring_collide': (40, 40, 40),
	'ring_with_obstacle': (40, 40, 40)
}

visualize_res = {
	'leapfrog': (128, 128, 128),
	'single_vortex_ring': (128, 128, 128),
	'ring_collide': (128, 128, 128),
	'ring_with_obstacle': (128, 128, 128)
}


'''
Other informations
'''

other_info = {
	'leapfrog': {
		'ring1': {
			'center': [.75, .5, .5],
			'normal': [-1., 0., 0.],
			'radius': 1. / 6,
			'thickness': .12 / 6,
			'strength': .1 / 6,
			'n': 500
		},
		'ring2': {
			'center': [.85, .5, .5],
			'normal': [-1., 0., 0.],
			'radius': .7 / 6,
			'thickness': .12 / 6,
			'strength': .1 / 6,
			'n': 500
		}
	},
	'single_vortex_ring': {
		'center': [.5, .5, .5],
		'normal': [1., 0., 0.],
		'radius': 1. / 6,
		'thickness': .1 / 6,
		'strength': .1 / 6,
		'n': 500
	},
	'ring_collide': {
		'ring1': {
			'center': [-.5 / 6 + .5, .5, .5],
			'normal': [1., 0., 0.],
			'radius': .3 / 6,
			'thickness': .12 / 6,
			'strength': .1 / 6,
			'n': 500
		},
		'ring2': {
			'center': [.5 / 6 + .5, .5, .5],
			'normal': [-1., 0., 0.],
			'radius': .3 / 6,
			'thickness': .12 / 6,
			'strength': .1 / 6,
			'n': 500
		}
	},
	'ring_with_obstacle': {
		'obj_file': '../assets/bunny.obj',
		'scale': 1. / 4.8, # 1. / 2.4,
		'rotate': torch.eye(3, device=device),
		'translate': torch.tensor([0.8225, 0.3150, 0.2650], device=device), # torch.tensor([1.17, .03, 0.], device=device),
		'rings': [
			{
				'center': [.475, .6, .53],
				'normal': [.2 / 1.08, .2 / 1.08, -1. / 1.08],
				'radius': .05,
				'thickness': .02,
				'strength': .2 / 6,
				'n': 500
			},
			{
				'center': [0.4380, 0.5630, 0.7152],
				'normal': [.2 / 1.08, .2 / 1.08, -1. / 1.08],
				'radius': .05,
				'thickness': .02,
				'strength': .2 / 6,
				'n': 500
			}
		]
	}
}


'''
Field and Jacobian
'''

@ti.func
def cross_matrix(a: tm.vec3):
	return tm.mat3([
		[0, -a[2], a[1]],
		[a[2], 0, -a[0]],
		[-a[1], a[0], 0]
	])
@ti.kernel
def vortex_particle(x: TiArr, x0: TiArr, w: TiArr, U: ti.f32, a: ti.f32, res: TiArr):
	for i in range(x.shape[0]):
		for j in range(x0.shape[0]):
			delta_p = tm.vec3([x[i, 0] - x0[j, 0], x[i, 1] - x0[j, 1], x[i, 2] - x0[j, 2]])
			r = tm.length(delta_p)
			fr = 1. / r ** 3 * (1. - tm.exp(-(r / a) ** 3))
			cur_term = U * fr * (cross_matrix(tm.vec3([w[j, 0], w[j, 1], w[j, 2]])) @ delta_p)
			res[i, 0] += cur_term[0]
			res[i, 1] += cur_term[1]
			res[i, 2] += cur_term[2]
@ti.kernel
def vortex_particle_gradient(x: TiArr, x0: TiArr, w: TiArr, U: ti.f32, a: ti.f32, res: TiArr):
	for i in range(x.shape[0]):
		for j in range(x0.shape[0]):
			delta_p = tm.vec3([x[i, 0] - x0[j, 0], x[i, 1] - x0[j, 1], x[i, 2] - x0[j, 2]])
			r = tm.length(delta_p)
			fr = 1. / r ** 3 * (1. - tm.exp(-(r / a) ** 3))
			fr_prime = -3. / r ** 4 * (1. - tm.exp(-(r / a) ** 3)) + 3. / (a ** 3 * r) * tm.exp(-(r / a) ** 3)
			cross_w = cross_matrix(tm.vec3([w[j, 0], w[j, 1], w[j, 2]]))
			cur_term = U * (fr_prime / r) * (cross_w @ delta_p.outer_product(delta_p)) + U * fr * cross_w
			for k in range(3):
				for l in range(3):
					res[i, k, l] += cur_term[k, l]

def vortex_ring(x, center, normal, radius, thickness, strength, n: int):
	axis_x = torch.tensor([1., 0., 0.], device=device)
	if torch.linalg.cross(axis_x, normal).norm() < 1e-5:
		axis_x = torch.tensor([0., 1., 0.], device=device)
	axis_y = torch.linalg.cross(normal, axis_x)
	axis_y /= axis_y.norm()
	axis_x = torch.linalg.cross(axis_y, normal)
	theta = torch.linspace(0., 2. * torch.pi, n + 1, device=device)[:-1]
	x0 = (axis_x[None, :] * torch.cos(theta)[:, None] + axis_y[None, :] * torch.sin(theta)[:, None]) * radius + center
	w = axis_x[None, :] * -torch.sin(theta)[:, None] + axis_y[None, :] * torch.cos(theta)[:, None]
	res = torch.zeros_like(x, device=device)
	vortex_particle(x, x0, w * strength, radius / (2 * n), thickness, res)
	return res
def vortex_ring_gradient(x, center, normal, radius, thickness, strength, n: int):
	axis_x = torch.tensor([1., 0., 0.], device=device)
	if torch.linalg.cross(axis_x, normal).norm() < 1e-5:
		axis_x = torch.tensor([0., 1., 0.], device=device)
	axis_y = torch.linalg.cross(normal, axis_x)
	axis_y /= axis_y.norm()
	axis_x = torch.linalg.cross(axis_y, normal)
	theta = torch.linspace(0., 2. * torch.pi, n + 1, device=device)[:-1]
	x0 = (axis_x[None, :] * torch.cos(theta)[:, None] + axis_y[None, :] * torch.sin(theta)[:, None]) * radius + center
	w = axis_x[None, :] * -torch.sin(theta)[:, None] + axis_y[None, :] * torch.cos(theta)[:, None]
	res = torch.zeros((x.shape[0], 3, 3), device=device)
	vortex_particle_gradient(x, x0, w * strength, radius / (2 * n), thickness, res)
	return res

def leapfrog(x):
	ring1 = other_info['leapfrog']['ring1']
	ring2 = other_info['leapfrog']['ring2']
	return vortex_ring(x, torch.tensor(ring1['center'], device=device), torch.tensor(ring1['normal'], device=device), ring1['radius'], ring1['thickness'], ring1['strength'], ring1['n'])\
		+ vortex_ring(x, torch.tensor(ring2['center'], device=device), torch.tensor(ring2['normal'], device=device), ring2['radius'], ring2['thickness'], ring2['strength'], ring2['n'])
def leapfrog_gradient(x):
	ring1 = other_info['leapfrog']['ring1']
	ring2 = other_info['leapfrog']['ring2']
	return vortex_ring_gradient(x, torch.tensor(ring1['center'], device=device), torch.tensor(ring1['normal'], device=device), ring1['radius'], ring1['thickness'], ring1['strength'], ring1['n'])\
		+ vortex_ring_gradient(x, torch.tensor(ring2['center'], device=device), torch.tensor(ring2['normal'], device=device), ring2['radius'], ring2['thickness'], ring2['strength'], ring2['n'])
leapfrog.gradient = leapfrog_gradient

def single_vortex_ring(x):
	ring = other_info['single_vortex_ring']
	return vortex_ring(x, torch.tensor(ring['center'], device=device), torch.tensor(ring['normal'], device=device), ring['radius'], ring['thickness'], ring['strength'], ring['n'])
def single_vortex_ring_gradeint(x):
	ring = other_info['single_vortex_ring']
	return vortex_ring_gradient(x, torch.tensor(ring['center'], device=device), torch.tensor(ring['normal'], device=device), ring['radius'], ring['thickness'], ring['strength'], ring['n'])
single_vortex_ring.gradient = single_vortex_ring_gradeint

def ring_collide(x):
	ring1 = other_info['ring_collide']['ring1']
	ring2 = other_info['ring_collide']['ring2']
	return vortex_ring(x, torch.tensor(ring1['center'], device=device), torch.tensor(ring1['normal'], device=device), ring1['radius'], ring1['thickness'], ring1['strength'], ring1['n'])\
		+ vortex_ring(x, torch.tensor(ring2['center'], device=device), torch.tensor(ring2['normal'], device=device), ring2['radius'], ring2['thickness'], ring2['strength'], ring2['n'])
def ring_collide_gradient(x):
	ring1 = other_info['ring_collide']['ring1']
	ring2 = other_info['ring_collide']['ring2']
	return vortex_ring_gradient(x, torch.tensor(ring1['center'], device=device), torch.tensor(ring1['normal'], device=device), ring1['radius'], ring1['thickness'], ring1['strength'], ring1['n'])\
		+ vortex_ring_gradient(x, torch.tensor(ring2['center'], device=device), torch.tensor(ring2['normal'], device=device), ring2['radius'], ring2['thickness'], ring2['strength'], ring2['n'])
ring_collide.gradient = ring_collide_gradient

def ring_with_obstacle(x):
	res = torch.zeros_like(x, device=device)
	for ring in other_info['ring_with_obstacle']['rings']:
		res += vortex_ring(x, torch.tensor(ring['center'], device=device), torch.tensor(ring['normal'], device=device), ring['radius'], ring['thickness'], ring['strength'], ring['n'])
	return res
def ring_with_obstacle_gradient(x):
	res = torch.zeros((x.shape[0], 3, 3), device=device)
	for ring in other_info['ring_with_obstacle']['rings']:
		res += vortex_ring_gradient(x, torch.tensor(ring['center'], device=device), torch.tensor(ring['normal'], device=device), ring['radius'], ring['thickness'], ring['strength'], ring['n'])
	return res
ring_with_obstacle.gradient = ring_with_obstacle_gradient


'''
Boundary sampler
'''

if 'obj_file' in other_info[cmd_args.init_cond]:
	obj_sampler = MeshSampler(other_info[cmd_args.init_cond]['obj_file'], other_info[cmd_args.init_cond]['scale'], other_info[cmd_args.init_cond]['rotate'], other_info[cmd_args.init_cond]['translate'])
	obj_sampler.save_obj(os.path.join(cmd_args.dir, 'obstacle.obj'))

def sample_on_box(n, x_min, x_max, y_min, y_max, z_min, z_max):
	x_scale, y_scale, z_scale = x_max - x_min, y_max - y_min, z_max - z_min
	t = torch.rand(n).to(device) * (x_scale * y_scale + y_scale * z_scale + z_scale * x_scale) * 2.
	data, normal = torch.zeros((n, 3), device=device), torch.zeros((n, 3), device=device)
	face0 = t < y_scale * z_scale
	face1 = torch.logical_and(y_scale * z_scale <= t, t < 2. * y_scale * z_scale)
	face2 = torch.logical_and(2. * y_scale * z_scale <= t, t < 2. * y_scale * z_scale + z_scale * x_scale)
	face3 = torch.logical_and(2. * y_scale * z_scale + z_scale * x_scale <= t, t < 2. * (y_scale * z_scale + z_scale * x_scale))
	face4 = torch.logical_and(2. * (y_scale * z_scale + z_scale * x_scale) <= t, t < 2. * (y_scale * z_scale + z_scale * x_scale) + x_scale * y_scale)
	face5 = 2. * (y_scale * z_scale + z_scale * x_scale) + x_scale * y_scale <= t
	data[face0, 0], data[face0, 1], data[face0, 2] = x_min, torch.rand_like(data[face0, 1], device=device) * y_scale + y_min, torch.rand_like(data[face0, 2], device=device) * z_scale + z_min
	data[face1, 0], data[face1, 1], data[face1, 2] = x_max, torch.rand_like(data[face1, 1], device=device) * y_scale + y_min, torch.rand_like(data[face1, 2], device=device) * z_scale + z_min
	data[face2, 0], data[face2, 1], data[face2, 2] = torch.rand_like(data[face2, 0], device=device) * x_scale + x_min, y_min, torch.rand_like(data[face2, 2], device=device) * z_scale + z_min
	data[face3, 0], data[face3, 1], data[face3, 2] = torch.rand_like(data[face3, 0], device=device) * x_scale + x_min, y_max, torch.rand_like(data[face3, 2], device=device) * z_scale + z_min
	data[face4, 0], data[face4, 1], data[face4, 2] = torch.rand_like(data[face4, 0], device=device) * x_scale + x_min, torch.rand_like(data[face4, 1], device=device) * y_scale + y_min, z_min
	data[face5, 0], data[face5, 1], data[face5, 2] = torch.rand_like(data[face5, 0], device=device) * x_scale + x_min, torch.rand_like(data[face5, 1], device=device) * y_scale + y_min, z_max
	normal[face0] = torch.tensor([1., 0., 0.], device=device)
	normal[face1] = torch.tensor([-1., 0., 0.], device=device)
	normal[face2] = torch.tensor([0., 1., 0.], device=device)
	normal[face3] = torch.tensor([0., -1., 0.], device=device)
	normal[face4] = torch.tensor([0., 0., 1.], device=device)
	normal[face5] = torch.tensor([0., 0., -1.], device=device)
	return data, normal

def sample_on_domain_boundary(n):
	x_min, x_max, y_min, y_max, z_min, z_max = domain[cmd_args.init_cond]
	return sample_on_box(n, x_min, x_max, y_min, y_max, z_min, z_max)

def sample_for_ring_with_obstacle(n):
	data1, normal1 = sample_on_domain_boundary(n)
	data2, normal2 = obj_sampler.sample(n)
	return torch.cat([data1, data2], dim=0), torch.cat([normal1, normal2], dim=0)

boundary_sampler = {
	'leapfrog': sample_on_domain_boundary,
	'single_vortex_ring': sample_on_domain_boundary,
	'ring_collide': sample_on_domain_boundary,
	'ring_with_obstacle': sample_for_ring_with_obstacle
}
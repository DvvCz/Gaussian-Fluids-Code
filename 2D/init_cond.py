import torch

from GSR import *


'''
Domain
'''

# [x_min, x_max, y_min, y_max]

initialize_domain = {
	'taylor_green': (0., 2. * np.pi, 0., 2. * np.pi),
	'taylor_vortex': (-5., 5., -5., 5.),
	'leapfrog': (-5., 5., -5., 5.),
	'vortices_pass': (0., 1., 0., 1.),
	'vortices_pass_narrow': (0., 1., 0., 1.),
	'vortices_pass_noslip': (0., 1., 0., 1.),
	'vortices_pass_particles': (-5., 5., -5., 5.),
	'karman': (-6.10321, 1.906778, -0.598466, 0.60349) # (-6. * np.pi, 2. * np.pi, -np.pi / 2., np.pi / 2.)
}
def get_scaling_factor():
	x_min, x_max, y_min, y_max = initialize_domain[cmd_args.init_cond]
	return 10. / min(x_max - x_min, y_max - y_min)
scaling_factor = get_scaling_factor()

initial_particle_count = {
	'taylor_green': (24, 24),
	'taylor_vortex': (71, 71),
	'leapfrog': (71, 71),
	'vortices_pass': (71, 71),
	'vortices_pass_narrow': (71, 71),
	'vortices_pass_noslip': (71, 71),
	'vortices_pass_particles': (71, 71),
	'karman': (400, 60) # (568, 71)
}

advance_domain = {
	'taylor_green': (0., 2. * np.pi, 0., 2. * np.pi),
	'taylor_vortex': (-5., 5., -5., 5.),
	'leapfrog': (-5., 5., -5., 5.),
	'vortices_pass': (0., 1., 0., 1.),
	'vortices_pass_narrow': (0., 1., 0., 1.),
	'vortices_pass_noslip': (0., 1., 0., 1.),
	'vortices_pass_particles': (-5., 5., -5., 5.),
	'karman': [initialize_domain['karman'][0], 1.906778, -0.598466, 0.60349] # , 2. * np.pi, -np.pi / 2., np.pi / 2.]
}

visualize_domain = {
	'taylor_green': (0., 2. * np.pi, 0., 2. * np.pi),
	'taylor_vortex': (-5., 5., -5., 5.),
	'leapfrog': (-5., 5., -5., 5.),
	'vortices_pass': (0., 1., 0., 1.),
	'vortices_pass_narrow': (0., 1., 0., 1.),
	'vortices_pass_noslip': (0., 1., 0., 1.),
	'vortices_pass_particles': (-2.5, 2.5, -2.5, 2.5),
	'karman': (-1.10321, 1.906778, -0.598466, 0.60349) # (0., 2. * np.pi, -np.pi / 2., np.pi / 2.)
}

visualize_res = {
	'taylor_green': (200, 200),
	'taylor_vortex': (200, 200),
	'leapfrog': (200, 200),
	'vortices_pass': (200, 200),
	'vortices_pass_narrow': (200, 200),
	'vortices_pass_noslip': (200, 200),
	'vortices_pass_particles': (200, 200),
	'karman': (501, 200)
}


'''
Other informations
'''

other_info = {
	'taylor_green': {},
	'taylor_vortex': {
		'U': 3.,
		'a': .5,
		'vortex_pos1': (-.8, 0.),
		'vortex_pos2': (.8, 0.)
	},
	'leapfrog': {
		'U': .5,
		'a': .3,
		'vortex_pos1': (-3., -3.),
		'vortex_pos2': (-1., -3.),
		'vortex_pos3': (1., -3.),
		'vortex_pos4': (3., -3.)
	},
	'vortices_pass': {
		'U': 5e-3,
		'a': 3e-2,
		'vortex_pos1': (.1, .525),
		'vortex_pos2': (.1, .475),
		'obstacle_pos1': (.5, .27),
		'obstacle_pos2': (.5, .73),
		'obstacle_radius': 60. / 511.
	},
	'vortices_pass_narrow': {
		'U': 5e-3,
		'a': 3e-2,
		'vortex_pos1': (.1, .525),
		'vortex_pos2': (.1, .475),
		'obstacle_pos1': (.5, .285),
		'obstacle_pos2': (.5, .715),
		'obstacle_radius': 60. / 511.
	},
	'vortices_pass_noslip': {
		'U': 5e-3,
		'a': 3e-2,
		'vortex_pos1': (.1, .525),
		'vortex_pos2': (.1, .475),
		'obstacle_pos1': (.5, .27),
		'obstacle_pos2': (.5, .73),
		'obstacle_radius': 60. / 511.
	},
	'vortices_pass_particles': {
		'particles_obj': '../assets/vortices_pass_particles.obj',
		'obstacle_pos1': (0., 1.),
		'obstacle_pos2': (0., -1.),
		'obstacle_radius': .25
	},
	'karman': {
		'v_magnitude': .5,
		'obstacle_pos': (-0.80356845, -0.00502235), # (np.pi / 4., .01),
		'obstacle_radius': 0.04553178393357534, # np.pi / 30.,
		'd0': np.pi / 15.
	}
}


'''
Field and jacobian
'''

def vortex_particle(x, x0, radius, magnitude, grad: bool):
	eps = 1e-6
	dx = x - x0
	r = (dx ** 2).sum(axis=-1) ** .5
	exp_term = torch.exp(-((r + eps) / radius) ** 2)
	if grad:
		part1 = torch.zeros((x.shape[0], 2, 2), device=device)
		part2 = torch.zeros((x.shape[0], 2, 2), device=device)
		part1[:, 0, 0] = dx[:, 0] * dx[:, 1]
		part1[:, 0, 1] = dx[:, 1] ** 2
		part1[:, 1, 0] = -dx[:, 0] ** 2
		part1[:, 1, 1] = -dx[:, 0] * dx[:, 1]
		part1 *= (2. * magnitude / r / (r + eps) * ((r + eps) ** -2. * (1. - exp_term) - radius ** -2. * exp_term))[:, None, None]
		part2[:, 0, 1] = -1.
		part2[:, 1, 0] = 1.
		part2 *= (magnitude * (r + eps) ** -2. * (1. - exp_term))[:, None, None]
		return part1 + part2
	else:
		return (magnitude * (r + eps) ** -2. * (1. - exp_term))[:, None] * torch.stack([-dx[:, 1], dx[:, 0]], dim=-1)

def taylor_green(x, grad: bool):
	if grad:
		g = torch.zeros((x.shape[0], 2, 2), device=device)
		g[:, 0, 0] = torch.cos(x[:, 0]) * torch.cos(x[:, 1])
		g[:, 0, 1] = -torch.sin(x[:, 0]) * torch.sin(x[:, 1])
		g[:, 1, 0] = torch.sin(x[:, 0]) * torch.sin(x[:, 1])
		g[:, 1, 1] = -torch.cos(x[:, 0]) * torch.cos(x[:, 1])
		return g
	else:
		return torch.stack([torch.sin(x[:, 0]) * torch.cos(x[:, 1]), -torch.cos(x[:, 0]) * torch.sin(x[:, 1])], dim=1)

def taylor_vortex(x, grad: bool):
	x_1, y_1 = other_info['taylor_vortex']['vortex_pos1']
	x_2, y_2 = other_info['taylor_vortex']['vortex_pos2']
	U, a = other_info['taylor_vortex']['U'], other_info['taylor_vortex']['a']
	r2_1, r2_2 = (x[:, 0] - x_1) ** 2 + (x[:, 1] - y_1) ** 2, (x[:, 0] - x_2) ** 2 + (x[:, 1] - y_2) ** 2
	if grad:
		res_1, res_2 = torch.zeros((x.shape[0], 2, 2), device=device), torch.zeros((x.shape[0], 2, 2), device=device)
		res_1[:, 0, 0] = (x_1 - x[:, 0]) * (y_1 - x[:, 1]) / a ** 2
		res_1[:, 0, 1] = (y_1 - x[:, 1]) ** 2 / a ** 2 - 1.
		res_1[:, 1, 0] = 1. - (x_1 - x[:, 0]) ** 2 / a ** 2
		res_1[:, 1, 1] = (x[:, 0] - x_1) * (y_1 - x[:, 1]) / a ** 2
		res_1 *= U / a * torch.exp(.5 * (1. - r2_1 / a ** 2))[:, None, None]
		res_2[:, 0, 0] = (x_2 - x[:, 0]) * (y_2 - x[:, 1]) / a ** 2
		res_2[:, 0, 1] = (y_2 - x[:, 1]) ** 2 / a ** 2 - 1.
		res_2[:, 1, 0] = 1. - (x_2 - x[:, 0]) ** 2 / a ** 2
		res_2[:, 1, 1] = (x[:, 0] - x_2) * (y_2 - x[:, 1]) / a ** 2
		res_2 *= U / a * torch.exp(.5 * (1. - r2_2 / a ** 2))[:, None, None]
		return res_1 + res_2
	else:
		return (
			torch.cat([y_1 - x[:, 1:], x[:, :1] - x_1], dim=1) * U / a * torch.exp(.5 * (1. - r2_1 / a ** 2))[:, None] +
			torch.cat([y_2 - x[:, 1:], x[:, :1] - x_2], dim=1) * U / a * torch.exp(.5 * (1. - r2_2 / a ** 2))[:, None]
		)

def leapfrog(x, grad: bool):
	x_1, y_1 = other_info['leapfrog']['vortex_pos1']
	x_2, y_2 = other_info['leapfrog']['vortex_pos2']
	x_3, y_3 = other_info['leapfrog']['vortex_pos3']
	x_4, y_4 = other_info['leapfrog']['vortex_pos4']
	U, a = other_info['leapfrog']['U'], other_info['leapfrog']['a']
	return vortex_particle(x, torch.tensor([x_1, y_1], device=device), a, U, grad)\
		+ vortex_particle(x, torch.tensor([x_2, y_2], device=device), a, U, grad)\
		+ vortex_particle(x, torch.tensor([x_3, y_3], device=device), a, -U, grad)\
		+ vortex_particle(x, torch.tensor([x_4, y_4], device=device), a, -U, grad)

def vortices_pass(x, grad: bool):
	x_1, y_1 = other_info[cmd_args.init_cond]['vortex_pos1']
	x_2, y_2 = other_info[cmd_args.init_cond]['vortex_pos2']
	U, a = other_info[cmd_args.init_cond]['U'], other_info[cmd_args.init_cond]['a']
	return vortex_particle(x, torch.tensor([x_1, y_1], device=device), a, U, grad)\
		+ vortex_particle(x, torch.tensor([x_2, y_2], device=device), a, -U, grad)
vortices_pass_narrow = vortices_pass
vortices_pass_noslip = vortices_pass

if cmd_args.init_cond == 'vortices_pass_particles':
	vort_particles_X, vort_particles_Y, vort_particles_W = [], [], []
	with open(other_info['vortices_pass_particles']['particles_obj'], 'r') as fd:
		for line in fd.readlines():
			if line.startswith('v '):
				vort_particles_X.append(float(line.split(' ')[1]))
				vort_particles_Y.append(float(line.split(' ')[3]))
				vort_particles_W.append(float(line.split(' ')[4]))
	
	vort_particles_pos = torch.tensor([vort_particles_X, vort_particles_Y], device=device).transpose(0, 1)
	vort_particles_strength = torch.tensor(vort_particles_W, device=device)
	
	def vortices_pass_particles_single(x):
		eps = .1
		delta_pos = vort_particles_pos - x[None, :]
		rescaled_sum = (vort_particles_strength[:, None] * delta_pos / ((delta_pos ** 2).sum(axis=-1)[:, None] + eps)).sum(axis=0)
		res = torch.zeros_like(x, device=device)
		res[0] = -rescaled_sum[1]
		res[1] = rescaled_sum[0]
		return res
	def vortices_pass_particles(x, grad: bool):
		if grad:
			return torch.vmap(torch.func.jacfwd(vortices_pass_particles_single))(x).contiguous()
		return torch.vmap(vortices_pass_particles_single)(x).contiguous()

# def karman_single(x):
# 	x0, y0 = other_info['karman']['obstacle_pos']
# 	r = other_info['karman']['obstacle_radius']
# 	d0 = other_info['karman']['d0']
# 	d = ((x[0] - x0) ** 2 + (x[1] - y0) ** 2) ** .5
# 	signed_d = (d - r) / d0
# 	ramp = torch.where(signed_d >= 1., 1., torch.where(signed_d <= 0., 0., 10. * signed_d ** 3 - 15. * signed_d ** 4 + 6. * signed_d ** 5))
# 	dramp = torch.where(torch.logical_or(signed_d >= 1., signed_d <= 0.), 0., 30. * signed_d ** 2 - 60. * signed_d ** 3 + 30. * signed_d ** 4)
# 	res = torch.zeros_like(x, device=device)
# 	res[0] += ramp
# 	res[0] += x[1] / (d * d0) * dramp * (x[1] - y0)
# 	res[1] += x[1] / (d * d0) * dramp * (x0 - x[0])
# 	res *= other_info['karman']['v_magnitude']
# 	return res
def karman_single(x):
	res = torch.zeros_like(x, device=device)
	res[0] += other_info['karman']['v_magnitude']
	return res

def karman(x, grad: bool):
	if grad:
		return torch.vmap(torch.func.jacfwd(karman_single))(x).contiguous()
	return torch.vmap(karman_single)(x).contiguous()


'''
Extra advector
'''

def karman_extra_advector(dt, advection_scheme='rk4'):
	advance_domain['karman'][0] = min(
		advance_domain['karman'][0] + dt * other_info['karman']['v_magnitude'],
		visualize_domain['karman'][0]
	)

extra_advector = {
	'taylor_green': None,
	'taylor_vortex': None,
	'leapfrog': None,
	'vortices_pass': None,
	'vortices_pass_narrow': None,
	'vortices_pass_noslip': None,
	'vortices_pass_particles': None,
	'karman': karman_extra_advector
}

def karman_extra_loader():
	advance_domain['karman'][0] = min(
		initialize_domain['karman'][0] + (cmd_args.start_frame * cmd_args.dt) * other_info['karman']['v_magnitude'],
		visualize_domain['karman'][0]
	)

extra_loader = {
	'taylor_green': None,
	'taylor_vortex': None,
	'leapfrog': None,
	'vortices_pass': None,
	'vortices_pass_narrow': None,
	'vortices_pass_noslip': None,
	'vortices_pass_particles': None,
	'karman': karman_extra_loader
}


'''
Boundary sampler
'''

def sample_on_domain_boundary_2(n):
	x_min, x_max, y_min, y_max = advance_domain[cmd_args.init_cond]
	x_scale, y_scale = x_max - x_min, y_max - y_min
	t = torch.rand(n).to(device) * (x_scale + y_scale) * 2.
	data, normal = torch.zeros((n, 2), device=device), torch.zeros((n, 2), device=device)
	edge0 = t < x_scale
	edge1 = torch.logical_and(x_scale <= t, t < x_scale + y_scale)
	edge2 = torch.logical_and(x_scale + y_scale <= t, t < 2. * x_scale + y_scale)
	edge3 = t >= 2. * x_scale + y_scale
	data[edge0, 0], data[edge0, 1] = x_min + t[edge0], y_min
	data[edge1, 0], data[edge1, 1] = x_max, y_min + t[edge1] - x_scale
	data[edge2, 0], data[edge2, 1] = x_max - t[edge2] + x_scale + y_scale, y_max
	data[edge3, 0], data[edge3, 1] = x_min, y_max - t[edge3] + 2. * x_scale + y_scale
	normal[edge0] = torch.tensor([0., -1.], device=device)
	normal[edge1] = torch.tensor([1., 0.], device=device)
	normal[edge2] = torch.tensor([0., 1.], device=device)
	normal[edge3] = torch.tensor([-1., 0.], device=device)
	return data, normal, torch.zeros(n, device=device)

def sample_on_sphere_1(n, x, y, r):
	theta = torch.rand(n).to(device) * 2. * np.pi
	data, value = torch.zeros((n, 2), device=device), torch.zeros((n, 2), device=device)
	data[:n, 0] = r * torch.cos(theta) + x
	data[:n, 1] = r * torch.sin(theta) + y
	return data, value

def sample_on_sphere_2(n, x, y, r):
	theta = torch.rand(n).to(device) * 2. * np.pi
	data, normal = torch.zeros((n, 2), device=device), torch.zeros((n, 2), device=device)
	data[:n, 0] = r * torch.cos(theta) + x
	data[:n, 1] = r * torch.sin(theta) + y
	normal[:n, 0] = torch.cos(theta)
	normal[:n, 1] = torch.sin(theta)
	return data, normal, torch.zeros(n, device=device)

def sample_for_vortices_pass_1(n):
	x1, y1 = other_info[cmd_args.init_cond]['obstacle_pos1']
	x2, y2 = other_info[cmd_args.init_cond]['obstacle_pos2']
	r = other_info[cmd_args.init_cond]['obstacle_radius']
	data1, value1 = sample_on_sphere_1(n, x1, y1, r)
	data2, value2 = sample_on_sphere_1(n, x2, y2, r)
	return torch.cat([data1, data2], dim=0), torch.cat([value1, value2], dim=0)

def sample_for_vortices_pass_2(n):
	x1, y1 = other_info[cmd_args.init_cond]['obstacle_pos1']
	x2, y2 = other_info[cmd_args.init_cond]['obstacle_pos2']
	r = other_info[cmd_args.init_cond]['obstacle_radius']
	data1, normal1, normal_val1 = sample_on_sphere_2(n, x1, y1, r)
	data2, normal2, normal_val2 = sample_on_sphere_2(n, x2, y2, r)
	data3, normal3, normal_val3 = sample_on_domain_boundary_2(n)
	return torch.cat([data1, data2, data3], dim=0), torch.cat([normal1, normal2, normal3], dim=0), torch.cat([normal_val1, normal_val2, normal_val3], dim=0)

def sample_for_vortices_pass_particles_2(n):
	x1, y1 = other_info[cmd_args.init_cond]['obstacle_pos1']
	x2, y2 = other_info[cmd_args.init_cond]['obstacle_pos2']
	r = other_info[cmd_args.init_cond]['obstacle_radius']
	data1, normal1, normal_val1 = sample_on_sphere_2(n, x1, y1, r)
	data2, normal2, normal_val2 = sample_on_sphere_2(n, x2, y2, r)
	return torch.cat([data1, data2], dim=0), torch.cat([normal1, normal2], dim=0), torch.cat([normal_val1, normal_val2], dim=0)

def sample_on_left_boundary_1(n):
	x_min, x_max, y_min, y_max = advance_domain['karman']
	data, value = torch.zeros((n, 2), device=device), torch.zeros((n, 2), device=device)
	data[:, 0] += x_min
	data[:, 1] = torch.rand(n, device=device) * (y_max - y_min) + y_min
	value[:, 0] += other_info['karman']['v_magnitude']
	return data, value

def sample_for_karman_1(n):
	return sample_on_sphere_1(n, other_info['karman']['obstacle_pos'][0], other_info['karman']['obstacle_pos'][1], other_info['karman']['obstacle_radius'])

def sample_for_karman_2(n):
	x_min, x_max, y_min, y_max = advance_domain['karman']
	x_min_v = visualize_domain['karman'][0]
	t = torch.rand(n).to(device) * (x_max - x_min) + x_min
	t2 = torch.rand(n).to(device) * (y_max - y_min) + y_min
	data, normal, normal_val = torch.zeros((n * 5, 2), device=device), torch.zeros((n * 5, 2), device=device), torch.zeros(n * 5, device=device)
	# Upper & Lower
	data[:n, 0] += t
	data[n:n * 2, 0] += t
	data[:n, 1] += y_min
	data[n:n * 2, 1] += y_max
	normal[:n, 1] += 1.
	normal[n:n * 2, 1] += -1.
	# Left
	data[n * 2:n * 3, 0] += x_min
	data[n * 2:n * 3, 1] += t2
	normal[n * 2:n * 3, 0] += 1.
	normal_val[n * 2:n * 3] += other_info['karman']['v_magnitude']
	# Right
	data[n * 3:n * 4, 0] += x_max
	data[n * 3:n * 4, 1] += t2
	normal[n * 3:n * 4, 0] += -1.
	normal_val[n * 3:n * 4] += -other_info['karman']['v_magnitude']
	# Left of visualize_domain
	data[n * 4:, 0] += x_min_v
	data[n * 4:, 1] += t2
	normal[n * 4:, 0] += 1.
	normal_val[n * 4:] += other_info['karman']['v_magnitude']
	return data, normal, normal_val

def target_boundary_sampler_1(boundary_sampler_1):
	def sample(n):
		data, value = boundary_sampler_1(n)
		return data * scaling_factor, value * scaling_factor
	return sample

def target_boundary_sampler_2(boundary_sampler_2):
	def sample(n):
		data, normal, normal_val = boundary_sampler_2(n)
		return data * scaling_factor, normal, normal_val * scaling_factor
	return sample

boundary_sampler = {
	'taylor_green': [None, target_boundary_sampler_2(sample_on_domain_boundary_2)],
	'taylor_vortex': [None, target_boundary_sampler_2(sample_on_domain_boundary_2)],
	'leapfrog': [None, target_boundary_sampler_2(sample_on_domain_boundary_2)],
	'vortices_pass': [None, target_boundary_sampler_2(sample_for_vortices_pass_2)],
	'vortices_pass_narrow': [None, target_boundary_sampler_2(sample_for_vortices_pass_2)],
	'vortices_pass_noslip': [target_boundary_sampler_1(sample_for_vortices_pass_1), target_boundary_sampler_2(sample_on_domain_boundary_2)],
	'vortices_pass_particles': [None, target_boundary_sampler_2(sample_for_vortices_pass_particles_2)],
	'karman': [target_boundary_sampler_1(sample_for_karman_1), target_boundary_sampler_2(sample_for_karman_2)]
}


'''
Field converter
'''

def target_field(original_field):
	def f(x):
		return scaling_factor * original_field(x / scaling_factor)
	return f

def target_gradient(original_gradient):
	def f(x):
		return original_gradient(x / scaling_factor)
	return f

def original_field(target_field):
	def g(x):
		return 1. / scaling_factor * target_field(x * scaling_factor)
	return g

def original_gradient(target_gradient):
	def g(x):
		return target_gradient(x * scaling_factor)
	return g
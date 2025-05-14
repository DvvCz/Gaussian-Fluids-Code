from init_cond import *

x_min, x_max, y_min, y_max, z_min, z_max = domain[cmd_args.init_cond]
x_N, y_N, z_N = visualize_res[cmd_args.init_cond]
x_N *= 4
y_N *= 4
z_N *= 4

@ti.func
def ti_get_coord(i, j, k, x_N, y_N, z_N, x_min, x_max, y_min, y_max, z_min, z_max):
	return tm.vec3([x_min + (x_max - x_min) / (x_N - 1) * i, y_min + (y_max - y_min) / (y_N - 1) * j, z_min + (z_max - z_min) / (z_N - 1) * k])

@ti.kernel
def ti_set_ring(density: TiArr, center: tm.vec3, normal: tm.vec3, radius: ti.f32, thickness: ti.f32, x_min: ti.f32, x_max: ti.f32, y_min: ti.f32, y_max: ti.f32, z_min: ti.f32, z_max: ti.f32):
	for i, j, k in density:
		pos = ti_get_coord(i, j, k, density.shape[0], density.shape[1], density.shape[2], x_min, x_max, y_min, y_max, z_min, z_max)
		pos_proj = pos - tm.dot(pos - center, normal) * normal
		if tm.length(pos_proj - center) >= radius - thickness:
			nearest_pos = center + tm.normalize(pos_proj - center) * radius
			if tm.length(pos - nearest_pos) <= thickness:
				density[i, j, k] = 1.

@ti.kernel
def ti_get_interp_val(field: TiArr, positions: TiArr, result: TiArr, x_min: ti.f32, x_max: ti.f32, y_min: ti.f32, y_max: ti.f32, z_min: ti.f32, z_max: ti.f32):
	dx = (x_max - x_min) / (field.shape[0] - 1)
	dy = (y_max - y_min) / (field.shape[1] - 1)
	dz = (z_max - z_min) / (field.shape[2] - 1)
	zero_p = tm.vec3([x_min, y_min, z_min])
	for i, j, k in result:
		p = tm.vec3([positions[i, j, k, 0], positions[i, j, k, 1], positions[i, j, k, 2]]) - zero_p
		pi = tm.floor(p[0] / dx, dtype=ti.i32)
		pj = tm.floor(p[1] / dy, dtype=ti.i32)
		pk = tm.floor(p[2] / dz, dtype=ti.i32)
		pi1 = ti.min(pi + 1, field.shape[0] - 1)
		pj1 = ti.min(pj + 1, field.shape[1] - 1)
		pk1 = ti.min(pk + 1, field.shape[2] - 1)
		corner_min = ti_get_coord(pi, pj, pk, field.shape[0], field.shape[1], field.shape[2], x_min, x_max, y_min, y_max, z_min, z_max) - zero_p
		w = p - corner_min
		w[0] /= dx
		w[1] /= dy
		w[2] /= dz
		result[i, j, k] =\
			field[pi, pj, pk] * (1. - w[0]) * (1. - w[1]) * (1. - w[2]) +\
			field[pi1, pj, pk] * w[0] * (1. - w[1]) * (1. - w[2]) +\
			field[pi, pj1, pk] * (1. - w[0]) * w[1] * (1. - w[2]) +\
			field[pi1, pj1, pk] * w[0] * w[1] * (1. - w[2]) +\
			field[pi, pj, pk1] * (1. - w[0]) * (1. - w[1]) * w[2] +\
			field[pi1, pj, pk1] * w[0] * (1. - w[1]) * w[2] +\
			field[pi, pj1, pk1] * (1. - w[0]) * w[1] * w[2] +\
			field[pi1, pj1, pk1] * w[0] * w[1] * w[2]

def advected_density(density, gaussian_velocity: GaussianSplatting3DFast, dt):
	x = get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_N, y_N, z_N)
	bk_x = gaussian_velocity.advection_rk4(x, -dt)
	bk_x.clamp_(torch.tensor([x_min, y_min, z_min], device=device), torch.tensor([x_max, y_max, z_max], device=device))
	bk_x = bk_x.reshape(x_N, y_N, z_N, 3)
	next_density = torch.zeros_like(density, device=device)
	ti_get_interp_val(density, bk_x, next_density, x_min, x_max, y_min, y_max, z_min, z_max)
	return next_density

def advected_density_N(density, gaussian_velocity: GaussianSplatting3DFast, dt, N):
	x = get_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, x_N, y_N, z_N)
	for i in range(N - 1, -1, -1):
		gaussian_velocity.load(os.path.join(cmd_args.dir, f'gaussian_velocity_{i}.pt'), False)
		v = gaussian_velocity(x)
		x -= v * dt
	x.clamp_(torch.tensor([x_min, y_min, z_min], device=device), torch.tensor([x_max, y_max, z_max], device=device))
	x = x.reshape(x_N, y_N, z_N, 3)
	next_density = torch.zeros_like(density, device=device)
	ti_get_interp_val(density, x, next_density, x_min, x_max, y_min, y_max, z_min, z_max)
	return next_density

def tensor2vti(V, x_min, x_max, y_min, y_max, z_min, z_max, save_filename: str):
	x_N, y_N, z_N = V.shape[0], V.shape[1], V.shape[2]
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

if __name__ == '__main__':
	if cmd_args.init_cond == 'ring_collide':
		density1 = torch.zeros((x_N, y_N, z_N), device=device)
		density2 = torch.zeros((x_N, y_N, z_N), device=device)
		ring1 = other_info[cmd_args.init_cond]['ring1']
		ring2 = other_info[cmd_args.init_cond]['ring2']
		ti_set_ring(density1, ring1['center'], ring1['normal'], ring1['radius'], ring1['thickness'] * 1., x_min, x_max, y_min, y_max, z_min, z_max)
		ti_set_ring(density2, ring2['center'], ring2['normal'], ring2['radius'], ring2['thickness'] * 1., x_min, x_max, y_min, y_max, z_min, z_max)
	else:
		raise NotImplementedError
	
	frame = 0
	if cmd_args.init_cond == 'ring_collide':
		tensor2vti(density1, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_a_{frame}.vti'))
		tensor2vti(density2, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_b_{frame}.vti'))
	gaussian_velocity = GaussianSplatting3DFast(x_min, x_max, y_min, y_max, z_min, z_max, np.zeros((1, 3)), dim=3, load_file=os.path.join(cmd_args.dir, f'gaussian_velocity_{frame}.pt'))
	while True:
		try:
			gaussian_velocity.load(os.path.join(cmd_args.dir, f'gaussian_velocity_{frame}.pt'), False)
		except:
			break
		frame += 1
		if cmd_args.init_cond == 'ring_collide':
			density1 = advected_density(density1, gaussian_velocity, cmd_args.dt)
			density2 = advected_density(density2, gaussian_velocity, cmd_args.dt)
			# density1 = (density1 - density1.min()) / (density1.max() - density1.min())
			# density2 = (density2 - density2.min()) / (density2.max() - density2.min())
			tensor2vti(density1, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_a_{frame}.vti'))
			tensor2vti(density2, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_b_{frame}.vti'))
			# cur_density1 = advected_density_N(density1, gaussian_velocity, cmd_args.dt, frame)
			# cur_density2 = advected_density_N(density2, gaussian_velocity, cmd_args.dt, frame)
			# tensor2vti(cur_density1, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_a_{frame}.vti'))
			# tensor2vti(cur_density2, x_min, x_max, y_min, y_max, z_min, z_max, os.path.join(cmd_args.dir, f'density_b_{frame}.vti'))
		print(f'Frame {frame} finished.')
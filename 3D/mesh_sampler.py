import torch
import taichi as ti

from GSR import *


@ti.data_oriented
class MeshSampler:
	def __init__(self, obj_file: str, scale, rotate, translate):
		self.load_obj(obj_file, scale, rotate, translate)
	
	@ti.kernel
	def ti_get_tri_area(self, vertices: TiArr, faces: TiArr, area_presum: TiArr):
		for i in range(faces.shape[0]):
			a = tm.vec3(vertices[faces[i, 0], 0], vertices[faces[i, 0], 1], vertices[faces[i, 0], 2])
			b = tm.vec3(vertices[faces[i, 1], 0], vertices[faces[i, 1], 1], vertices[faces[i, 1], 2])
			c = tm.vec3(vertices[faces[i, 2], 0], vertices[faces[i, 2], 1], vertices[faces[i, 2], 2])
			area_presum[i] = tm.length(tm.cross(b - a, c - a)) * .5
		for _ in range(1):
			for i in range(1, faces.shape[0]):
				area_presum[i] += area_presum[i-1]
	
	def load_obj(self, obj_file: str, scale, rotate, translate):
		self.vertices = []
		self.normals = []
		self.faces = []
		self.facenormals = []
		with open(obj_file, 'r') as fd:
			for line in fd.readlines():
				if line.startswith('v '):
					self.vertices.append(list(map(float, line.split(' ')[1:])))
				elif line.startswith('vn '):
					self.normals.append(list(map(float, line.split(' ')[1:])))
				elif line.startswith('f '):
					self.faces.append(list(map(lambda s: int(s.split('/')[0]) - 1, line.split(' ')[1:])))
					self.facenormals.append(list(map(lambda s: int(s.split('/')[-1]) - 1, line.split(' ')[1:])))
		self.vertices = (rotate[None] @ (scale * torch.tensor(self.vertices, device=device)).unsqueeze(-1)).squeeze(-1) + translate
		self.normals = (rotate[None] @ torch.tensor(self.normals, device=device).unsqueeze(-1)).squeeze(-1)
		self.normals /= ((self.normals ** 2.).sum(axis=-1) ** .5)[:, None]
		self.faces = torch.tensor(self.faces, dtype=torch.int32, device=device)
		self.facenormals = torch.tensor(self.facenormals, dtype=torch.int32, device=device)
		self.area_presum = torch.zeros(self.faces.shape[0], device=device)
		self.ti_get_tri_area(self.vertices, self.faces, self.area_presum)
		
		x_min, x_max = self.vertices[:, 0].min(), self.vertices[:, 0].max()
		y_min, y_max = self.vertices[:, 1].min(), self.vertices[:, 1].max()
		z_min, z_max = self.vertices[:, 2].min(), self.vertices[:, 2].max()
		print(f'Bounding box: [{x_min}, {x_max}] x [{y_min}, {y_max}] x [{z_min}, {z_max}]')
		print(f'Center: ({(x_min + x_max) * .5}, {(y_min + y_max) * .5}, {(z_min + z_max) * .5})')
	
	def save_obj(self, obj_file: str):
		with open(obj_file, 'w') as fd:
			for i in range(self.vertices.shape[0]):
				fd.write(f'v {self.vertices[i, 0].item()} {self.vertices[i, 1].item()} {self.vertices[i, 2].item()}\n')
			for i in range(self.normals.shape[0]):
				fd.write(f'vn {self.normals[i, 0].item()} {self.normals[i, 1].item()} {self.normals[i, 2].item()}\n')
			for i in range(self.faces.shape[0]):
				fd.write(f'f {self.faces[i, 0].item() + 1}//{self.facenormals[i, 0].item() + 1} {self.faces[i, 1].item() + 1}//{self.facenormals[i, 1].item() + 1} {self.faces[i, 2].item() + 1}//{self.facenormals[i, 2].item() + 1}\n')
	
	@ti.func
	def ti_lower_bound(self, t: ti.f32, arr: TiArr):
		l, r = 0, arr.shape[0]
		while l < r:
			m = (l + r) // 2
			if arr[m] < t:
				l = m + 1
			else:
				r = m
		return ti.min(l, arr.shape[0] - 1)
	
	@ti.kernel
	def ti_sample(self, N: ti.i32, vertices: TiArr, normals: TiArr, faces: TiArr, facenormals: TiArr, area_presum: TiArr, data: TiArr, normal: TiArr):
		total_area = area_presum[area_presum.shape[0]-1]
		for i in range(N):
			t = ti.random() * total_area
			face_id = self.ti_lower_bound(t, area_presum)
			u = 1. - ti.random() ** .5
			v = ti.random() * (1. - u)
			a = tm.vec3(vertices[faces[face_id, 0], 0], vertices[faces[face_id, 0], 1], vertices[faces[face_id, 0], 2])
			b = tm.vec3(vertices[faces[face_id, 1], 0], vertices[faces[face_id, 1], 1], vertices[faces[face_id, 1], 2])
			c = tm.vec3(vertices[faces[face_id, 2], 0], vertices[faces[face_id, 2], 1], vertices[faces[face_id, 2], 2])
			p = u * a + v * b + (1. - u - v) * c
			na = tm.vec3(normals[facenormals[face_id, 0], 0], normals[facenormals[face_id, 0], 1], normals[facenormals[face_id, 0], 2])
			nb = tm.vec3(normals[facenormals[face_id, 1], 0], normals[facenormals[face_id, 1], 1], normals[facenormals[face_id, 1], 2])
			nc = tm.vec3(normals[facenormals[face_id, 2], 0], normals[facenormals[face_id, 2], 1], normals[facenormals[face_id, 2], 2])
			n = tm.normalize(u * na + v * nb + (1. - u - v) * nc)
			data[i, 0], data[i, 1], data[i, 2] = p[0], p[1], p[2]
			normal[i, 0], normal[i, 1], normal[i, 2] = n[0], n[1], n[2]
	
	def sample(self, n):
		data = torch.zeros((n, 3), device=device)
		normal = torch.zeros((n, 3), device=device)
		self.ti_sample(n, self.vertices, self.normals, self.faces, self.facenormals, self.area_presum, data, normal)
		return data, normal


if __name__ == '__main__':
	# s = MeshSampler('../assets/bunny.obj', 1., torch.eye(3, device=device), torch.zeros(3, device=device))
	self = MeshSampler('../assets/bunny.obj', 1. / 2.4, torch.eye(3, device=device), torch.tensor([1.07, -.07, 1.01], device=device))
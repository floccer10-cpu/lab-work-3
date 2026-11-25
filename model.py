from pathlib import Path
import numpy as np
from numba import njit
from math import sqrt, pi
import tkinter as tk
from PIL import ImageTk, Image, ImageOps

ROTATION_STEP = pi/12
OFFSET_STEP = 1
LIGHT_DIR = np.array([1/3, 1/2, 1], dtype=np.float64)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
EPSILON = 0.1e-8



@njit(cache=True)
def bar_coord(x0, y0, z0, x1, y1, z1, x2, y2, z2, x, y):
    div = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if div == 0:
        return 1/3, 1/3, 1/3
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / div
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / div
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


@njit(cache=True)
def triangle(buf, z_buf, W, H,
             u0, u1, u2, v0, v1, v2,
             x0, y0, z0, x1, y1, z1, x2, y2, z2,
             normal0, normal1, normal2, light_dir, image):
    x_min = int(max(0, min(min(x0, x1), x2)))
    y_min = int(max(0, min(min(y0, y1), y2)))
    x_max = int(min(W, max(max(x0, x1), x2)))
    y_max = int(min(H, max(max(y0, y1), y2)))
    I0 = normal0[0] * light_dir[0] + normal0[1] * light_dir[1] +normal0[2] * light_dir[2]
    I1 = normal1[0] * light_dir[0] + normal1[1] * light_dir[1] +normal1[2] * light_dir[2]
    I2 = normal2[0] * light_dir[0] + normal2[1] * light_dir[1] +normal2[2] * light_dir[2]
    for _x in range(x_min, min(x_max + 1, W)):
        for _y in range(y_min, min(y_max + 1, H)):
            x = _x + 0.5
            y = _y + 0.5
            lambda0, lambda1, lambda2 = bar_coord(
                x0, y0, z0, x1, y1, z1, x2, y2, z2, x, y
            )
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buf[_y, _x]:
                    z_buf[_y, _x] = z
                    I = max(0, (lambda0 * I0 + lambda1 * I1 + lambda2 * I2) * (-1) )
                    img_x = image.shape[1] * (lambda0 * u0 + lambda1 * u1 + lambda2 * u2)
                    img_y = image.shape[0] * (lambda0 * v0 + lambda1 * v1 + lambda2 * v2)
                    r, g, b = image[int(img_y+0.5), int(img_x+0.5)]
                    buf[_y, image.shape[1] - 1 - _x] = (r*I, g*I, b*I)
                    

@njit(cache=True)
def render_model(buf, z_buf, vertices, vertices_normals, faces,
                 texture_vertices, texture_faces, image,
                 W, H, scale, light_dir, projection):
    for i in range(faces.shape[0]):
        face = faces[i]
        t_face = texture_faces[i]
        x0, y0, z0 = vertices[face[0]]
        x1, y1, z1 = vertices[face[1]]
        x2, y2, z2 = vertices[face[2]]

        ax = x1 - x2
        ay = y1 - y2
        az = z1 - z2
        bx = x1 - x0
        by = y1 - y0
        bz = z1 - z0
        
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
        norm = sqrt(nx * nx + ny * ny + nz * nz)
        if norm < EPSILON and dot < EPSILON or\
           norm > EPSILON and dot > EPSILON or\
           norm == 0.0 or dot == 0.0:
            continue

        cosine = dot / norm
        
        px0 = x0 * scale + W / 2.0
        py0 = -y0 * scale + H / 2.0
        px1 = x1 * scale + W / 2.0
        py1 = -y1 * scale + H / 2.0
        px2 = x2 * scale + W / 2.0
        py2 = -y2 * scale + H / 2.0

        if projection == "perspective":
            px0 = x0 * scale / z0 + W / 2.0
            px1 = x1 * scale / z1 + W / 2.0
            px2 = x2 * scale / z2 + W / 2.0
            py0 = y0 * scale / z0 + H / 2.0
            py1 = y1 * scale / z1 + H / 2.0
            py2 = y2 * scale / z2 + H / 2.0        

        u0, v0 = texture_vertices[t_face[0]]
        u1, v1 = texture_vertices[t_face[1]]
        u2, v2 = texture_vertices[t_face[2]]
        triangle(buf, z_buf, W, H,
                 u0, u1, u2, v0, v1, v2,
                 px0, py0, z0, px1, py1, z1, px2, py2, z2,
                 vertices_normals[face[0]],
                 vertices_normals[face[1]],
                 vertices_normals[face[2]],
                 light_dir, image)



class Model:
    def __init__(self, path, texture, scale):
        p = self._validate_path(path)
        self._parse_file(p)
        self._faces_to_edges()
        self._faces_to_normals()
        self._vertices_to_normals()
        self.image = None
        self._parse_image(texture)
        self.light_dir = LIGHT_DIR
        self.vertices[:] *= [scale, scale, scale]

    def _parse_image(self, file):
        try:
            with open(file) as f:
                img = ImageOps.flip(Image.open(file))
                self.image = np.asarray(img)
        except:
            pass
            

    def _faces_to_edges(self):
        self.edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [0, 2]]
        ], axis=0)
        self.edges = np.sort(self.edges, axis=1)
        self.edges = np.unique(self.edges, axis=0)

    def _faces_to_normals(self):
        self.normals = np.zeros(shape = self.faces.shape, dtype = np.float64)
        for i, face in enumerate(self.faces, start = 0):
            first = self.vertices[face[0]]
            second = self.vertices[face[1]]
            third = self.vertices[face[2]]

            self.normals[i] = np.cross(second - first, third - first)
            self.normals[i] /= np.linalg.norm(self.normals[i])  

    def _vertices_to_normals(self):
        self.vertices_normals = np.zeros(shape = self.vertices.shape, dtype = np.float64)
        for i, vertice in enumerate(self.vertices, start = 0):
            normal_sum = np.zeros(shape = 3, dtype = np.float64)
            faces = np.where(self.faces == i)[0]
            for face_idx in faces:
                normal_sum += self.normals[face_idx]
            self.vertices_normals[i] = normal_sum / faces.shape[0]

    @staticmethod
    def _validate_path(path):
        if not str(path).strip():
            raise ValueError("path can't be empty")
        p = Path(path)
        if not p.exists():
            raise ValueError("file doesn't exist")
        if not p.is_file():
            raise IsADirectoryError("not a file")
        suf = p.suffix
        if suf.lower() != ".obj":
            raise ValueError(f"expected .obj file, got {suf}")
        return p

    def _parse_file(self, p):
        with p.open(mode='r', errors="replace", encoding="utf-8") as f:
            V = []
            VT = []
            F = []
            FT = []
            for lineno, raw in enumerate(f, start=1):
                parts = raw.split()
                if len(parts) == 0:
                    continue
                if parts[0] == 'v':
                    if len(parts) < 4:
                        raise ValueError(
                            f"{p}:{lineno}: vertex must have 3 coordinates"
                        )
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        V.append((x, y, z))
                    except ValueError as e:
                        raise ValueError(
                            f"{p}:{lineno}: bad vertex: {e}"
                        ) from e
                elif parts[0] == 'vt':
                    if len(parts) < 3:
                        raise ValueError(
                            f"{p}:{lineno}: vertex texture must have 2 coordinates"
                        )
                    try:
                        x, y = float(parts[1]), float(parts[2])
                        VT.append((x, y))
                    except ValueError as e:
                        raise ValueError(
                            f"{p}:{lineno}: bad vertex texture: {e}"
                        ) from e
                elif parts[0] == 'f':
                    idx = []
                    t_idx = []
                    for token in parts[1:]:
                        splitted_token = token.split("/")
                        head = splitted_token[0]
                        texture_head = splitted_token[1]
                        if not head:
                            raise ValueError(
                                f"{p}:{lineno}: empty face index"
                            )
                        try:
                            idx.append(int(head) - 1)
                            t_idx.append(int(texture_head) - 1)
                        except ValueError as e:
                            raise ValueError(
                                f"{p}:{lineno}: bad face index: {e}"
                            ) from e
                    if len(idx) < 3:
                        raise ValueError(
                            f"{p}:{lineno}: face must point to at least 3 indices"
                        )
                    if any(i < 0 for i in idx):
                        raise ValueError(
                            f"{p}:{lineno}: face indices must be natural numbers"
                        )
                    for i in range(1, len(idx) - 1):
                        F.append((idx[0], idx[i], idx[i + 1]))
                    for i in range(1, len(t_idx) - 1):
                        FT.append((t_idx[0], t_idx[i], t_idx[i + 1]))
            if not V:
                raise ValueError(f"{p}: no vertices found")
            if not F:
                raise ValueError(f"{p}: no faces found")

            self.vertices = np.asarray(V, dtype=np.float64)
            self.faces = np.asarray(F, dtype=np.uint32)
            self.texture_vertices = np.asarray(VT, dtype=np.float64)
            self.texture_faces = np.asarray(FT, dtype=np.uint32)
            if self.faces.max() >= self.vertices.shape[0]:
                raise ValueError(
                    f"{p}: faces reference vertex {self.faces.max() + 1}, "
                    f"only {self.vertices.shape[0]} vertices defined"
                )


class Renderer:
    def __init__(self, W, H, scale):
        self.W = W
        self.H = H
        self.scale = scale
        self.xOffset = 0
        self.yOffset = 0.06*scale
        self.zOffset = 0.2*scale
        self.xRotation = 0
        self.yRotation = pi
        self.zRotation = pi
        self.img = np.zeros((H, W, 3), dtype=np.uint8)
        self.zbuf = np.full((H, W), np.inf, dtype=np.float64)
        self.projection = 'perspective'

    def render(self, model):
        self.img.fill(0)
        self.zbuf.fill(np.inf)
        transformed_vertices = model.vertices.copy()
        transformed_vertices_normals = model.vertices_normals.copy()

        if any(r != 0 for r in (self.xRotation, self.yRotation, self.zRotation)):
            cx, sx = np.cos(self.xRotation), np.sin(self.xRotation)
            cy, sy = np.cos(self.yRotation), np.sin(self.yRotation)
            cz, sz = np.cos(self.zRotation), np.sin(self.zRotation)
            
            Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
            
            R = Rx @ Ry @ Rz
            transformed_vertices = model.vertices @ R.T
            transformed_vertices_normals = model.vertices_normals @ R.T
        
        transformed_vertices[:] += [self.xOffset, self.yOffset, self.zOffset]

        render_model(
            self.img, self.zbuf, transformed_vertices, transformed_vertices_normals,
            model.faces, model.texture_vertices, model.texture_faces,
            model.image, self.W, self.H,
            self.scale if self.projection == "perspective" else 1,
            model.light_dir, projection = self.projection
        )

class App:
    def __init__(self, model, renderer, out_path="img.png",
                 win_w=900, win_h=900):
        self.model = model
        self.renderer = renderer
        self.out_path = out_path

        self.root = tk.Tk()
        self.root.geometry(f"{win_w}x{win_h}")

        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.focus_set()

        self.canvas.bind('<Configure>', lambda e: self._refit())
        self.root.bind('<Key>', self.on_key)

        self.photo = None
        self.full_image = None

        self.render()

    def on_key(self, event):
        k = event.keysym.lower()
        print(k, self.renderer.xOffset, self.renderer.yOffset, self.renderer.zOffset)
        if k == 'space':
            Image.fromarray(self.renderer.img).save(self.out_path)
            return 'break'
        elif k == 'r':
            self.renderer.projection = "perspective" if \
                                       self.renderer.projection == "parallel" \
                                       else "parallel"
        # camera movement
        elif k == 'up':
            self.model.vertices -= [0, 1/sqrt(self.renderer.scale), 0]
        elif k == 'down':
            self.model.vertices += [0, 1/sqrt(self.renderer.scale), 0]
        elif k == 'left':
            self.model.vertices += [1/sqrt(self.renderer.scale), 0, 0]
        elif k == 'right':
            self.model.vertices -= [1/sqrt(self.renderer.scale), 0, 0]
            
        # render scale
        elif k == 'plus' or k == 'equal':
            self.renderer.scale *= 1.1
        elif k == 'minus':
            self.renderer.scale /= 1.1

        # model rotation
        elif k == 'w':
            self.renderer.xRotation += ROTATION_STEP
        elif k == 's':
            self.renderer.xRotation -= ROTATION_STEP
        elif k == 'a':
            self.renderer.yRotation += ROTATION_STEP
        elif k == 'd':
            self.renderer.yRotation -= ROTATION_STEP
        elif k == 'q':
            self.renderer.zRotation += ROTATION_STEP
        elif k == 'e':
            self.renderer.zRotation -= ROTATION_STEP

        # model offset
        elif k == '6': #left
            self.renderer.xOffset += OFFSET_STEP*sqrt(self.renderer.scale)
        elif k == '4': #right
            self.renderer.xOffset -= OFFSET_STEP*sqrt(self.renderer.scale)
        elif k == '8': #up
            self.renderer.yOffset += OFFSET_STEP*sqrt(self.renderer.scale)
        elif k == '2': #down
            self.renderer.yOffset -= OFFSET_STEP*sqrt(self.renderer.scale)
        elif k == '5': #front
            self.renderer.zOffset -= OFFSET_STEP*sqrt(self.renderer.scale)
        elif k == '0': #back
            self.renderer.zOffset += OFFSET_STEP*sqrt(self.renderer.scale)
        else:
            return
        self.render()
        return 'break'

    def render(self):
        self.renderer.render(self.model)
        self.full_image = Image.fromarray(self.renderer.img)
        self._refit()

    def _refit(self):
        if self.full_image is None:
            return
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        if ww < 10 or wh < 10:
            return
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)

    def run(self):
        self.root.mainloop()

from pathlib import Path
import numpy as np
from numba import njit
from math import sqrt
import tkinter as tk
from PIL import ImageTk, Image

COLOR_MIN = 128
COLOR_MAX = 256
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
def triangle(buf, z_buf, W, H, r, g, b,
             x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x_min = int(max(0, min(min(x0, x1), x2)))
    y_min = int(max(0, min(min(y0, y1), y2)))
    x_max = int(min(W, max(max(x0, x1), x2)))
    y_max = int(min(H, max(max(y0, y1), y2)))

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
                    buf[_y, _x] = (r, g, b)
                    z_buf[_y, _x] = z

@njit(cache=True)
def render_model(buf, z_buf, vertices, faces, colors,
                 W, H, scale, light_dir):
    for i in range(faces.shape[0]):
        first, second, third = faces[i]

        x0, y0, z0 = vertices[first]
        x1, y1, z1 = vertices[second]
        x2, y2, z2 = vertices[third]

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
           norm > EPSILON and dot > EPSILON or norm == 0.0 or dot == 0.0:
            continue

        cosine = dot / norm

        px0 = x0 * scale + W / 2.0
        py0 = -y0 * scale + H / 2.0
        px1 = x1 * scale + W / 2.0
        py1 = -y1 * scale + H / 2.0
        px2 = x2 * scale + W / 2.0
        py2 = -y2 * scale + H / 2.0

        intensity = -cosine
        r, g, b = colors[i] * intensity

        r = min(r, 255)
        g = min(g, 255)
        b = min(b, 255)

        triangle(buf, z_buf, W, H, r, g, b,
                 px0, py0, z0, px1, py1, z1, px2, py2, z2)


class Model:
    def __init__(self, path):
        p = self._validate_path(path)
        self._parse_file(p)
        self._faces_to_edges()
        '''self.colors = np.random.randint(
            COLOR_MIN, COLOR_MAX, size=(len(self.faces), 3), dtype=np.uint8
        )'''
        self.colors = np.full((len(self.faces), 3), 255, dtype=np.uint8)
        self.light_dir = LIGHT_DIR

    def _faces_to_edges(self):
        self.edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [0, 2]]
        ], axis=0)
        self.edges = np.sort(self.edges, axis=1)
        self.edges = np.unique(self.edges, axis=0)

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
            F = []
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
                elif parts[0] == 'f':
                    idx = []
                    for token in parts[1:]:
                        head = token.split("/", 1)[0]
                        if not head:
                            raise ValueError(
                                f"{p}:{lineno}: empty face index"
                            )
                        try:
                            idx.append(int(head) - 1)
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
            if not V:
                raise ValueError(f"{p}: no vertices found")
            if not F:
                raise ValueError(f"{p}: no faces found")

            self.vertices = np.asarray(V, dtype=np.float64)
            self.faces = np.asarray(F, dtype=np.uint32)
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
        self.img = np.zeros((H, W, 3), dtype=np.uint8)
        self.zbuf = np.full((H, W), np.inf, dtype=np.float64)

    def render(self, model):
        self.img.fill(0)
        self.zbuf.fill(np.inf)
        render_model(
            self.img, self.zbuf,
            model.vertices, model.faces, model.colors,
            self.W, self.H, self.scale, model.light_dir
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
        k = event.keysym
        if k == 'space':
            Image.fromarray(self.renderer.img).save(self.out_path)
            return 'break'
        elif k == 'plus':
            self.renderer.scale *= 1.1
        elif k == 'minus':
            self.renderer.scale /= 1.1
        elif k == 'Up':
            self.model.vertices -= [0, 1/sqrt(self.renderer.scale), 0]
        elif k == 'Down':
            self.model.vertices += [0, 1/sqrt(self.renderer.scale), 0]
        elif k == 'Left':
            self.model.vertices += [1/sqrt(self.renderer.scale), 0, 0]
        elif k == 'Right':
            self.model.vertices -= [1/sqrt(self.renderer.scale), 0, 0]
            
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

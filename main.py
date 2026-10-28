from model import Model, Renderer, App

def main(w: int = 2560, h: int = 2560, s: float = 1024.0,
         src: str = "bunny.obj", out: str = "img.png") -> int:
    model = Model(src)
    renderer = Renderer(w, h, s)
    App(model, renderer, out_path=out).run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

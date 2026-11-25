from model import Model, Renderer, App

def main(w: int = 2560, h: int = 2560, s: float = 1024.0,
         src: str = "bunny.obj", txt: str = "bunny2.jpg", out: str = "img.png") -> int:
    model = Model(src, txt, s)
    renderer = Renderer(w, h, s)
    App(model, renderer, out_path=out).run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

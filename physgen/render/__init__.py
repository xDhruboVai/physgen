"""Renderer backends (doc D1). All consume the same scene config, poses and light track."""


def get_renderer(cfg, lib):
    name = cfg["renderer"]["backend"]
    if name == "raster":
        from .raster import RasterRenderer
        return RasterRenderer(cfg, lib)
    if name == "blender":
        from .blender import BlenderRenderer
        return BlenderRenderer(cfg, lib)
    raise ValueError(f"unknown renderer backend {name!r}")

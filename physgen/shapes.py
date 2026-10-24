"""Object geometry shared by physics and both renderers (one source of truth, doc 6.1).

A geometry is a plain dict stored in the scene config:
  sphere {radius} | box/brick/domino {half} | cylinder/capsule {radius, height} (axis z)
  wedge/hull {points} (convex, centred at the solid centroid, principal axes = body axes)
  mesh {mesh, scale} (loaded through assets.load_mesh)
"""
import numpy as np
from scipy.spatial import ConvexHull

BOXES = {"box", "brick", "domino"}
CONVEX = {"wedge", "hull"}
ROUND = {"sphere", "cylinder", "capsule"}


def mass_props(tris):
    """Volume, centroid and inertia tensor (unit density, about the centroid) of a closed triangle mesh."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = v.sum()
    com = ((a + b + c) / 4.0 * v[:, None]).sum(0) / vol
    s = a + b + c
    cov = sum(np.einsum("i,ij,ik->jk", v, x, x) for x in (a, b, c, s)) / 20.0
    cov -= vol * np.outer(com, com)
    return vol, com, np.trace(cov) * np.eye(3) - cov


def hull_tris(points):
    h = ConvexHull(points)
    tris = points[h.simplices]
    centre = points.mean(0)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", n, tris[:, 0] - centre) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def canonical_convex(points):
    """Centre convex points at the solid centroid and rotate to principal axes.
    Returns (points_body, upright_quat) where upright_quat maps body frame -> original frame."""
    points = np.asarray(points, float)
    _, com, inertia = mass_props(hull_tris(points))
    w, R = np.linalg.eigh(inertia)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    body = (points - com) @ R
    return body, mat_to_quat(R)


def wedge_points(half):
    hx, hy, hz = half
    return np.array([[x, y, -hz] for x in (-hx, hx) for y in (-hy, hy)] +
                    [[-hx, y, hz] for y in (-hy, hy)], float)


def volume(geom):
    s = geom["shape"]
    if s == "sphere":
        return 4 / 3 * np.pi * geom["radius"] ** 3
    if s in BOXES:
        return 8 * np.prod(geom["half"])
    if s == "cylinder":
        return np.pi * geom["radius"] ** 2 * geom["height"]
    if s == "capsule":
        r = geom["radius"]
        return np.pi * r * r * geom["height"] + 4 / 3 * np.pi * r ** 3
    if s in CONVEX:
        return mass_props(hull_tris(np.asarray(geom["points"])))[0]
    raise ValueError(s)


def convex_inertia(geom, mass):
    """Principal inertia diagonal for convex shapes (PyBullet only approximates meshes by their AABB)."""
    vol, _, inertia = mass_props(hull_tris(np.asarray(geom["points"])))
    return np.diag(inertia) * mass / vol


def mesh_arrays(geom, mesh_lib=None):
    """Triangle soup for rendering: positions (N,3), normals (N,3), uvs (N,2), all float32."""
    s = geom["shape"]
    if s == "sphere":
        p, n = _sphere(geom["radius"])
    elif s in BOXES:
        hx, hy, hz = geom["half"]
        p, n = _flat(hull_tris(np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])))
    elif s in CONVEX:
        p, n = _flat(hull_tris(np.asarray(geom["points"], float)))
    elif s == "cylinder":
        p, n = _cylinder(geom["radius"], geom["height"])
    elif s == "capsule":
        p, n = _capsule(geom["radius"], geom["height"])
    elif s == "mesh":
        m = mesh_lib.load(geom["mesh"])
        sc = geom["scale"]
        return (m["positions"] * sc).astype(np.float32), m["normals"].astype(np.float32), m["uvs"].astype(np.float32)
    else:
        raise ValueError(s)
    return p.astype(np.float32), n.astype(np.float32), np.zeros((len(p), 2), np.float32)


def local_points(geom, mesh_lib=None):
    """Vertices in the body frame (for support height, bounds and visibility)."""
    return mesh_arrays(geom, mesh_lib)[0]


def _flat(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return tris.reshape(-1, 3), np.repeat(n, 3, axis=0)


def _grid_tris(P, N):
    """Quad grid (rows, cols, 3) -> triangle soup positions and normals."""
    i, j = np.meshgrid(np.arange(P.shape[0] - 1), np.arange(P.shape[1] - 1), indexing="ij")
    quads = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j), (i + 1, j + 1), (i, j + 1)]
    idx = np.stack([np.stack(q, -1) for q in quads], 2).reshape(-1, 2)
    return P[idx[:, 0], idx[:, 1]], N[idx[:, 0], idx[:, 1]]


def _sphere(r, nu=32, nv=16):
    th = np.linspace(0, np.pi, nv + 1)[:, None]
    ph = np.linspace(0, 2 * np.pi, nu + 1)[None, :]
    N = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), -1)
    return _grid_tris(N * r, N)


def _cylinder(r, h, nu=32):
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    ring = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], -1)
    P = np.stack([ring * r + [0, 0, h / 2], ring * r - [0, 0, h / 2]])
    side_p, side_n = _grid_tris(P, np.stack([ring, ring]))
    caps_p, caps_n = [], []
    for z, sgn in ((h / 2, 1), (-h / 2, -1)):
        for k in range(nu):
            a, b = ring[k] * r + [0, 0, z], ring[k + 1] * r + [0, 0, z]
            tri = [[0, 0, z], a, b] if sgn > 0 else [[0, 0, z], b, a]
            caps_p += tri
            caps_n += [[0, 0, sgn]] * 3
    return np.concatenate([side_p, caps_p]), np.concatenate([side_n, caps_n])


def _capsule(r, h, nu=32, nv=16):
    th = np.linspace(0, np.pi, nv + 1)
    th = np.concatenate([th[: nv // 2 + 1], th[nv // 2:]])[:, None]  # duplicate equator ring for the cylinder band
    ph = np.linspace(0, 2 * np.pi, nu + 1)[None, :]
    N = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), -1)
    off = np.where(np.arange(len(th))[:, None, None] <= nv // 2, h / 2, -h / 2)
    P = N * r + off * np.array([0, 0, 1.0])
    return _grid_tris(P, N)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_mats(q):
    """Batched (..., 4) xyzw -> (..., 3, 3)."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def quat_mul(a, b):
    from scipy.spatial.transform import Rotation
    return (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()

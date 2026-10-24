"""Simulation runner (doc 8.2 'sim'). Backends implement a small set of operations that the c2
injectors use, so another engine (e.g. MuJoCo) only has to provide this class.

Body ids are scene object ids; the floor is FLOOR (-1).
"""
import numpy as np

from . import shapes

FLOOR = -1


class Trajectory:
    """Poses at every physics step: pos (N, K+1, 3), quat (N, K+1, 4) xyzw; velocities per step;
    contact pairs per step and contact-begin events."""

    def __init__(self, n_obj, n_steps):
        self.pos = np.zeros((n_obj, n_steps + 1, 3))
        self.quat = np.zeros((n_obj, n_steps + 1, 4))
        self.lin = np.zeros((n_obj, n_steps + 1, 3))
        self.ang = np.zeros((n_obj, n_steps + 1, 3))
        self.contacts = [frozenset()] * (n_steps + 1)  # set of (i, j) with i < j (floor = -1)
        self.events = []  # dicts: step, a, b, normal_speed
        self.overlap = np.zeros(n_steps + 1, bool)  # filled by injectors that track a pair
        self.max_penetration = 0.0
        self.unstable = False


class PyBulletBackend:
    name = "pybullet"

    def __init__(self, scene, mesh_lib=None):
        import pybullet as p
        self.p = p
        self.cid = p.connect(p.DIRECT)
        self.scene = scene
        self.mesh_lib = mesh_lib
        self.hz = scene["physics"]["hz"]

    def close(self):
        self.p.disconnect(self.cid)

    # ---------- world construction ----------
    def build(self):
        p, c = self.p, self.cid
        p.resetSimulation(physicsClientId=c)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / self.hz, numSolverIterations=self.scene["physics"]["solver_iterations"],
                                    numSubSteps=self.scene["physics"].get("internal_substeps", 1),
                                    deterministicOverlappingPairs=1, physicsClientId=c)
        p.setGravity(0, 0, self.scene["physics"]["gravity"], physicsClientId=c)
        env = self.scene["environment"]
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=c)
        self.floor = p.createMultiBody(0, plane, physicsClientId=c)
        p.changeDynamics(self.floor, -1, lateralFriction=env["floor_friction"], restitution=env["floor_restitution"],
                         physicsClientId=c)
        self.bid, self.oid, self.inertia = {}, {self.floor: FLOOR}, {}
        for o in self.scene["objects"]:
            col = self._collision_shape(o["geom"])
            mass = o["mass_kg"] if o["role"] == "dynamic" else 0.0
            b = p.createMultiBody(mass, col, basePosition=o["pos"], baseOrientation=o["quat"], physicsClientId=c)
            kw = dict(lateralFriction=o["friction"], restitution=o["restitution"],
                      rollingFriction=o.get("rolling_friction", 0.0), spinningFriction=o.get("spinning_friction", 0.0),
                      linearDamping=0.0, angularDamping=0.0, physicsClientId=c)
            if o["role"] == "dynamic" and o["geom"]["shape"] in shapes.CONVEX:
                kw["localInertiaDiagonal"] = shapes.convex_inertia(o["geom"], mass).tolist()
            if o["role"] == "dynamic" and o.get("ccd"):
                kw["ccdSweptSphereRadius"] = o["ccd"]
            p.changeDynamics(b, -1, **kw)
            if o["role"] == "dynamic":
                p.resetBaseVelocity(b, o["lin_vel"], o["ang_vel"], physicsClientId=c)
            self.bid[o["id"]] = b
            self.oid[b] = o["id"]
            self.inertia[o["id"]] = np.array(p.getDynamicsInfo(b, -1, physicsClientId=c)[2])
        self.dynamic = [o["id"] for o in self.scene["objects"] if o["role"] == "dynamic"]
        self.kinematic = [o for o in self.scene["objects"] if o["role"] == "kinematic"]

    def _collision_shape(self, g):
        p, c = self.p, self.cid
        s = g["shape"]
        if s == "sphere":
            return p.createCollisionShape(p.GEOM_SPHERE, radius=g["radius"], physicsClientId=c)
        if s in shapes.BOXES:
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=g["half"], physicsClientId=c)
        if s == "cylinder":
            return p.createCollisionShape(p.GEOM_CYLINDER, radius=g["radius"], height=g["height"], physicsClientId=c)
        if s == "capsule":
            return p.createCollisionShape(p.GEOM_CAPSULE, radius=g["radius"], height=g["height"], physicsClientId=c)
        if s in shapes.CONVEX:
            return p.createCollisionShape(p.GEOM_MESH, vertices=np.asarray(g["points"]).tolist(), physicsClientId=c)
        if s == "mesh":
            m = self.mesh_lib.load(g["mesh"])
            return p.createCollisionShape(p.GEOM_MESH, fileName=m["collision_file"], meshScale=[g["scale"]] * 3,
                                          physicsClientId=c)
        raise ValueError(s)

    # ---------- operations used by injectors ----------
    def state(self, i):
        p, b = self.p, self.bid[i]
        pos, q = p.getBasePositionAndOrientation(b, physicsClientId=self.cid)
        lin, ang = p.getBaseVelocity(b, physicsClientId=self.cid)
        return np.array(pos), np.array(q), np.array(lin), np.array(ang)

    def set_velocity(self, i, lin=None, ang=None):
        _, _, l0, a0 = self.state(i)
        self.p.resetBaseVelocity(self.bid[i], (l0 if lin is None else lin).tolist(), (a0 if ang is None else ang).tolist(),
                                 physicsClientId=self.cid)

    def apply_force(self, i, force):
        # applied as a velocity change over this step: pybullet drops most of applyExternalForce when numSubSteps > 1
        _, _, lin, ang = self.state(i)
        self.set_velocity(i, lin + np.asarray(force) / self.mass(i) / self.hz, ang)

    def set_collision(self, i, j, enabled):
        bj = self.floor if j == FLOOR else self.bid[j]
        self.p.setCollisionFilterPair(self.bid[i], bj, -1, -1, int(enabled), physicsClientId=self.cid)

    def mass(self, i):
        return self.p.getDynamicsInfo(self.bid[i], -1, physicsClientId=self.cid)[0]

    def set_mass(self, i, m):
        info = self.p.getDynamicsInfo(self.bid[i], -1, physicsClientId=self.cid)
        scale = m / info[0]
        self.p.changeDynamics(self.bid[i], -1, mass=m, localInertiaDiagonal=[x * scale for x in info[2]],
                              physicsClientId=self.cid)

    def friction(self, i):
        return self.p.getDynamicsInfo(self.bid[i], -1, physicsClientId=self.cid)[1]

    def set_friction(self, i, mu):
        self.p.changeDynamics(self.bid[i], -1, lateralFriction=mu, physicsClientId=self.cid)

    def overlapping(self, i, j, tol=1e-3):
        if j == FLOOR:  # the plane is a half-space: overlap means straddling z = 0, not merely being below it
            lo, hi = self.p.getAABB(self.bid[i], physicsClientId=self.cid)
            return lo[2] < -tol and hi[2] > 0
        bj = self.bid[j]
        pts = self.p.getClosestPoints(self.bid[i], bj, 0.0, physicsClientId=self.cid)
        return any(pt[8] < -tol for pt in pts)

    def contact_points(self):
        """[(i, j, normal_on_j_pointing_to_i, distance, normal_force)]"""
        out = []
        for pt in self.p.getContactPoints(physicsClientId=self.cid):
            a, b = self.oid.get(pt[1]), self.oid.get(pt[2])
            if a is None or b is None:
                continue
            out.append((a, b, np.array(pt[7]), pt[8], pt[9]))
        return out

    def set_kinematic_pose(self, i, pos, quat):
        self.p.resetBasePositionAndOrientation(self.bid[i], list(pos), list(quat), physicsClientId=self.cid)

    def step(self):
        self.p.stepSimulation(physicsClientId=self.cid)


BACKENDS = {"pybullet": PyBulletBackend}


def kinematic_pose(o, step, steps_per_frame):
    """Pose of a scripted (kinematic) object at a step: linear move of `delta` over the frame window."""
    k = o["kinematic"]
    t = (step / steps_per_frame - k["start_frame"]) / max(k["duration_frames"], 1e-6)
    a = float(np.clip(t, 0, 1))
    return np.asarray(o["pos"]) + a * np.asarray(k["delta"]), o["quat"]


def simulate(scene, n_frames, hooks=(), mesh_lib=None, backend=None):
    """Run the scene for n_frames (plus nothing else). hooks: callables h(sim, step, traj) called
    before every step; they may also be called with phase='after' via h.after(sim, step, traj)."""
    S = scene["clip"]["substeps"]
    n_steps = n_frames * S
    own = backend is None
    sim = backend or BACKENDS[scene["physics"]["engine"]](scene, mesh_lib)
    sim.build()
    ids = [o["id"] for o in scene["objects"]]
    tr = Trajectory(len(ids), n_steps)
    tr.inertia = sim.inertia
    index = {i: k for k, i in enumerate(ids)}
    sim.index = index

    def record(k):
        for i in ids:
            pos, q, lin, ang = sim.state(i)
            n = index[i]
            tr.pos[n, k], tr.quat[n, k], tr.lin[n, k], tr.ang[n, k] = pos, q, lin, ang

    record(0)
    dyn = set(sim.dynamic)
    prev_pairs = frozenset()
    for k in range(n_steps):
        for o in sim.kinematic:
            pos, q = kinematic_pose(o, k + 1, S)
            sim.set_kinematic_pose(o["id"], pos, q)
        for h in hooks:
            h(sim, k, tr)
        sim.step()
        record(k + 1)
        pairs = set()
        for a, b, n, dist, f in sim.contact_points():
            if a == b or (a not in dyn and b not in dyn):
                continue
            pairs.add((min(a, b), max(a, b)))
            tr.max_penetration = max(tr.max_penetration, -dist)
        pairs = frozenset(pairs)
        for a, b in pairs - prev_pairs:
            na, nb = index.get(a), index.get(b)
            va = tr.lin[na, k] if na is not None else np.zeros(3)
            vb = tr.lin[nb, k] if nb is not None else np.zeros(3)
            tr.events.append(dict(step=k + 1, a=a, b=b, speed=float(np.linalg.norm(va - vb))))
        tr.contacts[k + 1] = pairs
        prev_pairs = pairs
        for h in hooks:
            if hasattr(h, "after"):
                h.after(sim, k, tr)
        v = np.abs(tr.lin[:, k + 1]).max()
        if not np.isfinite(tr.pos[:, k + 1]).all() or v > 25.0:
            tr.unstable = True
            break
    if own:
        sim.close()
    return tr

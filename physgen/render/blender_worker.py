"""Runs inside Blender: blender -b --factory-startup -P blender_worker.py -- job.json clip.npz out.npy

Builds the scene from renderer-agnostic data and renders linear radiance (tone mapping happens in
the driver, identically to the raster backend).

Two logical lights (doc 6.1): every object has a shadow-only proxy driven by the shadow poses;
visible objects cast no shadows. In Cycles, light groups give the shading light without shadows
(key), the shadow light with shadows (shadow) and without (unshadow); the compositor outputs
ambient + key * (shadow / unshadow), which equals ordinary shading when both lights coincide (c0).
EEVEE has no light groups, so there the unshadowed copy is a negative light instead (approximate
for c1 direction edits).
"""
import json
import sys

import bpy
import numpy as np


def main(job_path, clip_path, out_path):
    with open(job_path) as f:
        job = json.load(f)
    clip = dict(np.load(clip_path))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    r = job["render"]
    cycles = r["engine"] == "CYCLES"
    sc.render.engine = r["engine"]
    sc.render.resolution_x, sc.render.resolution_y = job["resolution"]
    sc.render.resolution_percentage = 100
    sc.render.fps = job["fps"]
    if cycles:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if r["device"] != "CPU":
            prefs.compute_device_type = r["device"]
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == r["device"]
            sc.cycles.device = "GPU"
        sc.cycles.samples = r["samples"]
        sc.cycles.use_denoising = r["denoise"]
        sc.cycles.seed = 0
        sc.cycles.use_animated_seed = False
    shutter, blur = job["shutter"], job["blur_samples"] > 1
    sc.render.use_motion_blur = blur
    sc.render.motion_blur_shutter = shutter
    if cycles:
        sc.cycles.motion_blur_position = "START"
    elif hasattr(sc.eevee, "motion_blur_position"):
        sc.eevee.motion_blur_position = "START"
    vl = sc.view_layers[0]

    S = job["substeps"]
    k0 = int(clip["k0"])
    times = (k0 + np.arange(clip["pos"].shape[1])) / S  # frame time of every stored step

    def keyframes(obj, path, values, interp=1):
        """values (K, n) at `times`; interp 1 = LINEAR, 0 = CONSTANT."""
        if obj.animation_data is None:
            obj.animation_data_create()
        if obj.animation_data.action is None:
            obj.animation_data.action = bpy.data.actions.new(obj.name)
        for c in range(values.shape[1]):
            fc = obj.animation_data.action.fcurves.new(path, index=c)
            fc.keyframe_points.add(len(values))
            fc.keyframe_points.foreach_set("co", np.c_[times[:len(values)], values[:, c]].ravel().tolist())
            fc.keyframe_points.foreach_set("interpolation", [interp] * len(values))
            fc.update()

    # ---------------- world (HDRI ambient)
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1, -1, 1)  # match the raster equirect convention
    mp.inputs["Rotation"].default_value = (0, 0, job["hdri_rot"])
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(job["hdri"])
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type, tint.blend_type = "RGBA", "MULTIPLY"
    tint.inputs[0].default_value = 1.0
    tint.inputs[7].default_value = (*job["light_color"], 1)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = job["world_strength"]
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], tint.inputs[6])
    nt.links.new(tint.outputs[2], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if cycles:
        vl.lightgroups.add(name="amb")
        world.lightgroup = "amb"

    images = {}

    def image(path):
        if path not in images:
            images[path] = bpy.data.images.load(path)
        return images[path]

    def material(name, m):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        t = mat.node_tree
        bsdf = t.nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = m["roughness"]
        bsdf.inputs["Metallic"].default_value = m["metallic"]
        tc = t.nodes.new("ShaderNodeTexCoord")
        mp = t.nodes.new("ShaderNodeMapping")
        tex = t.nodes.new("ShaderNodeTexImage")
        tex.image = image(m["texture"])
        if m["mode"] == "uv":
            t.links.new(tc.outputs["UV"], tex.inputs["Vector"])
        else:
            mp.inputs["Scale"].default_value = (1 / m["tex_scale"],) * 3
            t.links.new(tc.outputs["Object"], mp.inputs["Vector"])
            t.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
            if m["mode"] == "triplanar":
                tex.projection, tex.projection_blend = "BOX", 0.25
        con = t.nodes.new("ShaderNodeMix")  # mean + contrast * (tex - mean), then gain * tint
        con.data_type = "RGBA"
        con.inputs[0].default_value = m["contrast"]
        con.inputs[6].default_value = (*m["mean"], 1)
        t.links.new(tex.outputs["Color"], con.inputs[7])
        gain = t.nodes.new("ShaderNodeMix")
        gain.data_type, gain.blend_type = "RGBA", "MULTIPLY"
        gain.inputs[0].default_value = 1.0
        gain.inputs[7].default_value = (*[c * m["gain"] for c in m["tint"]], 1)
        t.links.new(con.outputs[2], gain.inputs[6])
        t.links.new(gain.outputs[2], bsdf.inputs["Base Color"])
        return mat

    def mesh_object(name, geo, mat):
        p = geo["positions"]
        me = bpy.data.meshes.new(name)
        n = len(p) // 3
        me.from_pydata(p.tolist(), [], [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(n)])
        me.polygons.foreach_set("use_smooth", [True] * n)
        uv = me.uv_layers.new()
        uv.data.foreach_set("uv", geo["uvs"].ravel().tolist())
        me.normals_split_custom_set_from_vertices(geo["normals"].tolist())
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        return ob

    # ---------------- floor
    fl = job["floor"]
    half = 20.0
    cx, cy = fl["centre"]
    floor_geo = dict(positions=np.array([[cx - half, cy - half, 0], [cx + half, cy - half, 0], [cx + half, cy + half, 0],
                                         [cx - half, cy - half, 0], [cx + half, cy + half, 0], [cx - half, cy + half, 0]]),
                     normals=np.tile([0, 0, 1.0], (6, 1)), uvs=np.zeros((6, 2)))
    mesh_object("floor", floor_geo, material("floor", fl))

    # ---------------- objects and their shadow proxies
    for k, o in enumerate(job["objects"]):
        geo = dict(np.load(o["geometry"]))
        vis = mesh_object(f"obj{k}", geo, material(f"mat{k}", o))
        prx = bpy.data.objects.new(f"proxy{k}", vis.data)
        sc.collection.objects.link(prx)
        vis.visible_shadow = False
        for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"):
            setattr(prx, attr, False)
        for ob, pos, quat in ((vis, clip["pos"][k], clip["quat"][k]), (prx, clip["spos"][k], clip["squat"][k])):
            ob.rotation_mode = "QUATERNION"
            keyframes(ob, "location", pos)
            keyframes(ob, "rotation_quaternion", quat[:, [3, 0, 1, 2]])

    # ---------------- lights: shading light (no shadow), shadow light, unshadowed copy of the shadow light
    def sun(name, group, energy, shadow):
        L = bpy.data.lights.new(name, "SUN")
        L.energy, L.color, L.use_shadow = energy, job["light_color"], shadow
        ob = bpy.data.objects.new(name, L)
        sc.collection.objects.link(ob)
        if cycles:
            ob.lightgroup = group
        return ob

    frames = clip["frames"].tolist()
    E = job["light_energy"]
    key = sun("key", "key", E, False)
    sh = sun("shadow", "shadow", E, True)
    un = sun("unshadow", "unshadow", E if cycles else -E, False)
    if cycles:
        for g in ("key", "shadow", "unshadow"):
            vl.lightgroups.add(name=g)

    def aim(ob, d, f):
        from mathutils import Matrix
        z = np.asarray(d, float) / np.linalg.norm(d)  # a sun shines along local -Z, so local +Z points at the light
        x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.999 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = x, np.cross(z, x), z
        ob.matrix_world = Matrix(M.tolist())
        ob.keyframe_insert("rotation_euler", frame=f)

    for f in frames:
        aim(key, clip["shade_dir"][f], f)
        for ob in (sh, un):
            aim(ob, clip["shadow_dir"][f], f)
            ob.data.angle = float(clip["sun_angle"][f])
            ob.data.keyframe_insert("angle", frame=f)
    for ob in (key, sh, un):
        for data in (ob.animation_data, ob.data.animation_data):
            if data and data.action:
                for fc in data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"

    # ---------------- camera
    cam = bpy.data.cameras.new("cam")
    cam.sensor_fit = "VERTICAL"
    cam.angle = np.radians(job["fov_deg"])
    cam.clip_start, cam.clip_end = 0.02, 60.0
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    co.rotation_mode = "QUATERNION"
    keyframes(co, "location", clip["cam_eye"])
    keyframes(co, "rotation_quaternion", clip["cam_quat"])

    # ---------------- compositor: linear radiance to the Viewer node
    if cycles and r["denoise"]:
        vl.cycles.denoising_store_passes = True
    sc.use_nodes = True
    ct = sc.node_tree
    ct.nodes.clear()
    rl = ct.nodes.new("CompositorNodeRLayers")
    viewer = ct.nodes.new("CompositorNodeViewer")
    comp = ct.nodes.new("CompositorNodeComposite")
    if cycles:
        def mix(kind, a, b, clamp=False):
            n = ct.nodes.new("CompositorNodeMixRGB")
            n.blend_type, n.use_clamp = kind, clamp
            n.inputs[0].default_value = 1.0
            for slot, src in ((1, a), (2, b)):
                if isinstance(src, tuple):
                    n.inputs[slot].default_value = src
                else:
                    ct.links.new(src, n.inputs[slot])
            return n.outputs[0]
        def denoised(name):  # the render denoiser only touches Combined, so denoise each light-group pass
            if not r["denoise"]:
                return rl.outputs[name]
            dn = ct.nodes.new("CompositorNodeDenoise")
            ct.links.new(rl.outputs[name], dn.inputs["Image"])
            ct.links.new(rl.outputs["Denoising Normal"], dn.inputs["Normal"])
            ct.links.new(rl.outputs["Denoising Albedo"], dn.inputs["Albedo"])
            return dn.outputs[0]
        eps = (1e-4, 1e-4, 1e-4, 1)
        vis_ratio = mix("DIVIDE", mix("ADD", denoised("Combined_shadow"), eps),
                        mix("ADD", denoised("Combined_unshadow"), eps), clamp=True)
        result = mix("ADD", denoised("Combined_amb"), mix("MULTIPLY", denoised("Combined_key"), vis_ratio))
    else:
        result = rl.outputs["Image"]
    ct.links.new(result, viewer.inputs[0])
    ct.links.new(result, comp.inputs[0])
    viewer.use_alpha = False

    # the Viewer node is not evaluated in background renders: write linear EXR and read it back
    import os
    W, H = job["resolution"]
    s = sc.render.image_settings
    s.file_format, s.color_depth, s.color_mode = "OPEN_EXR", "32", "RGB"
    out = np.zeros((len(frames), H, W, 4), np.float32)
    buf = np.zeros(W * H * 4, np.float32)
    exr = os.path.splitext(out_path)[0] + "_frame.exr"
    for n, f in enumerate(frames):
        sc.frame_set(f)
        sc.render.filepath = exr
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(exr)
        img.pixels.foreach_get(buf)
        out[n] = buf.reshape(H, W, 4)
        bpy.data.images.remove(img)
    os.remove(exr)
    np.save(out_path, out[:, ::-1, :, :3])


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    main(*argv)

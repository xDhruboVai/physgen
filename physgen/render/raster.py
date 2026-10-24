"""GPU rasterizer (moderngl / OpenGL 4.3) with its own shadow-map pass (doc 6.1, D1 option B).

Two logical lights: the shading light (diffuse + highlights) and the shadow light (direction,
softness and shadow-caster poses). c0/c2/c3 use identical lights; c1 decouples them.
Soft shadows use PCSS so blur grows with occluder height (S1.6 switches that relation).
"""
import collections

import moderngl
import numpy as np

from .. import camera, shapes
from . import common

_rng = np.random.default_rng(7)
_disk = []
while len(_disk) < 32:  # fixed Poisson-like disk (rejection sampled once, deterministic)
    p = _rng.uniform(-1, 1, 2)
    if np.hypot(*p) <= 1 and all(np.hypot(*(p - q)) > 0.22 for q in _disk):
        _disk.append(p)
POISSON = ",".join(f"vec2({x:.5f},{y:.5f})" for x, y in _disk)

VS = """
#version 430
uniform mat4 vp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
out vec3 w_pos; out vec3 w_nrm; out vec3 o_pos; out vec3 o_nrm; out vec2 v_uv;
void main(){
  vec4 wp = model * vec4(in_pos, 1.0);
  w_pos = wp.xyz; w_nrm = mat3(model) * in_nrm; o_pos = in_pos; o_nrm = in_nrm; v_uv = in_uv;
  gl_Position = vp * wp;
}
"""

FS = """
#version 430
in vec3 w_pos; in vec3 w_nrm; in vec3 o_pos; in vec3 o_nrm; in vec2 v_uv;
out vec4 frag;
uniform int tex_mode;            // 0 triplanar in object space, 1 mesh uv, 2 world planar (floor)
uniform sampler2D albedo_tex; uniform float tex_scale; uniform vec3 tint;
uniform vec3 tex_mean; uniform float tex_contrast; uniform float tex_gain;
uniform float roughness; uniform float metallic;
uniform vec3 cam_pos;
uniform vec3 shade_dir; uniform vec3 key_rgb; uniform float ambient;
uniform mat4 shadow_vp; uniform sampler2D shadow_map;
uniform float shadow_width; uniform float shadow_depth; uniform float light_tan;
uniform int pen_mode; uniform float shadow_texel;
uniform vec3 sh[9]; uniform sampler2D env; uniform float env_rot; uniform float env_lods;
uniform vec4 occ[24]; uniform int n_occ; uniform int self_occ;
uniform float fog_start; uniform float fog_end;
const vec2 disk[32] = vec2[](%POISSON%);
const float PI = 3.14159265;

vec2 eq_uv(vec3 d){
  float u = atan(d.y, d.x) / (2.0*PI) + 0.5 - env_rot / (2.0*PI);
  return vec2(fract(u), acos(clamp(d.z, -1.0, 1.0)) / PI);
}
vec3 sh_irr(vec3 n){
  float c = cos(-env_rot), s = sin(-env_rot);
  n = vec3(c*n.x - s*n.y, s*n.x + c*n.y, n.z);
  return sh[0]*0.282095 + sh[1]*0.488603*n.y + sh[2]*0.488603*n.z + sh[3]*0.488603*n.x
       + sh[4]*1.092548*n.x*n.y + sh[5]*1.092548*n.y*n.z + sh[6]*0.315392*(3.0*n.z*n.z-1.0)
       + sh[7]*1.092548*n.x*n.z + sh[8]*0.546274*(n.x*n.x-n.y*n.y);
}
vec3 albedo(){
  vec3 c;
  if (tex_mode == 2) c = texture(albedo_tex, w_pos.xy / tex_scale).rgb;
  else if (tex_mode == 1) c = texture(albedo_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;  // OBJ uv origin is bottom-left
  else {
    vec3 w = pow(abs(normalize(o_nrm)), vec3(4.0)); w /= (w.x + w.y + w.z);
    c = texture(albedo_tex, o_pos.yz / tex_scale).rgb * w.x + texture(albedo_tex, o_pos.xz / tex_scale).rgb * w.y
      + texture(albedo_tex, o_pos.xy / tex_scale).rgb * w.z;
  }
  return max((tex_mean + tex_contrast * (c - tex_mean)) * tex_gain, 0.0) * tint;
}
float shadow(vec3 p, vec3 n){
  vec4 s = shadow_vp * vec4(p + n * 2.0 * shadow_texel, 1.0);
  vec3 c = s.xyz / s.w * 0.5 + 0.5;
  if (any(lessThan(c, vec3(0.0))) || any(greaterThan(c, vec3(1.0)))) return 1.0;
  float ang = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  mat2 R = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float bias = 0.0015 / shadow_depth;
  float search = clamp(light_tan * 0.6 / shadow_width, 2.0 * shadow_texel / shadow_width, 0.05);
  float zb = 0.0; float nb = 0.0;
  for (int i = 0; i < 16; i++){
    float d = texture(shadow_map, c.xy + R * disk[i] * search).r;
    if (d < c.z - bias){ zb += d; nb += 1.0; }
  }
  if (nb == 0.0) return 1.0;
  float h = (c.z - zb / nb) * shadow_depth;          // receiver-occluder distance (m)
  float pen = 2.0 * light_tan * h;                    // penumbra width (m)
  if (pen_mode == 1) pen = 2.0 * light_tan * 0.08;   // S1.6: blur does not grow with height
  if (pen_mode == 2) pen = 2.0 * light_tan * max(0.4 - h, 0.0);  // S1.6: grows the wrong way
  float r = max(pen / shadow_width, 1.5 * shadow_texel / shadow_width);
  float lit = 0.0;
  for (int i = 0; i < 32; i++){
    float d = texture(shadow_map, c.xy + R * disk[i] * r).r;
    lit += (d >= c.z - bias) ? 1.0 : 0.0;
  }
  return lit / 32.0;
}
float ao(vec3 p, vec3 n){
  float a = 1.0;
  for (int i = 0; i < n_occ; i++){
    if (i == self_occ) continue;
    vec3 d = occ[i].xyz - p; float l = length(d);
    if (l > 4.0 * occ[i].w) continue;  // local only: distant objects must not change every pixel
    float o = clamp(occ[i].w * occ[i].w / (l * l) * max(dot(n, d / l), 0.0), 0.0, 1.0) * smoothstep(4.0, 3.0, l / occ[i].w);
    a *= 1.0 - 0.8 * o;
  }
  return a;
}
void main(){
  vec3 n = normalize(w_nrm);
  vec3 v = normalize(cam_pos - w_pos);
  if (dot(n, v) < 0.0 && tex_mode != 2) n = -n;
  vec3 a = albedo();
  float nl = max(dot(n, shade_dir), 0.0);
  float vis = shadow(w_pos, n);
  float occl = ao(w_pos, n);
  float rough = max(roughness, 0.05); float al = rough * rough;
  vec3 f0 = mix(vec3(0.04), a, metallic);
  vec3 hv = normalize(shade_dir + v);
  float nh = max(dot(n, hv), 0.0), nv = max(dot(n, v), 1e-3);
  float D = al * al / (PI * pow(nh * nh * (al * al - 1.0) + 1.0, 2.0));
  float k = (rough + 1.0) * (rough + 1.0) / 8.0;
  float G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
  vec3 F = f0 + (1.0 - f0) * pow(1.0 - max(dot(hv, v), 0.0), 5.0);
  vec3 spec = D * G * F / max(4.0 * nl * nv, 1e-3) * key_rgb * nl * vis;
  vec3 diff = (1.0 - metallic) * a / PI * (key_rgb * nl * vis + ambient * key_rgb * max(sh_irr(n), 0.0) * occl);
  vec3 r = reflect(-v, n);
  vec3 Fa = f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(1.0 - nv, 5.0);
  vec3 envspec = Fa * textureLod(env, eq_uv(r), rough * env_lods).rgb * ambient * key_rgb * occl;
  vec3 col = diff + spec + envspec;
  float fog = smoothstep(fog_start, fog_end, length(w_pos - cam_pos));
  col = mix(col, textureLod(env, eq_uv(-v), 1.0).rgb * ambient * key_rgb, fog);
  frag = vec4(col, 1.0);
}
""".replace("%POISSON%", POISSON)

SHADOW_VS = """
#version 430
uniform mat4 vp; uniform mat4 model;
in vec3 in_pos;
void main(){ gl_Position = vp * model * vec4(in_pos, 1.0); }
"""
SHADOW_FS = "#version 430\nvoid main(){}\n"

QUAD_VS = """
#version 430
out vec2 ndc;
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
  ndc = p; gl_Position = vec4(p, 1.0, 1.0);
}
"""
SKY_FS = """
#version 430
in vec2 ndc; out vec4 frag;
uniform mat4 inv_vp; uniform vec3 cam_pos; uniform sampler2D env; uniform float env_rot; uniform vec3 scale;
const float PI = 3.14159265;
void main(){
  vec4 p = inv_vp * vec4(ndc, 1.0, 1.0);
  vec3 d = normalize(p.xyz / p.w - cam_pos);
  float u = atan(d.y, d.x) / (2.0*PI) + 0.5 - env_rot / (2.0*PI);
  frag = vec4(textureLod(env, vec2(fract(u), acos(clamp(d.z,-1.0,1.0)) / PI), 1.0).rgb * scale, 1.0);
}
"""
ACC_FS = """
#version 430
in vec2 ndc; out vec4 frag;
uniform sampler2D src; uniform float weight;
void main(){ frag = vec4(texture(src, ndc * 0.5 + 0.5).rgb * weight, 1.0); }
"""


class RasterRenderer:
    name = "raster"

    def __init__(self, cfg, lib):
        self.cfg, self.lib = cfg, lib
        self.ctx = moderngl.create_standalone_context(require=430)
        c = self.ctx
        self.prog = c.program(vertex_shader=VS, fragment_shader=FS)
        self.sprog = c.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.sky = c.program(vertex_shader=QUAD_VS, fragment_shader=SKY_FS)
        self.acc = c.program(vertex_shader=QUAD_VS, fragment_shader=ACC_FS)
        self.empty_vao_sky = c.vertex_array(self.sky, [])
        self.empty_vao_acc = c.vertex_array(self.acc, [])
        n = cfg["renderer"]["shadow_map"]
        self.shadow_tex = c.depth_texture((n, n))
        self.shadow_tex.compare_func = ""
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = c.framebuffer(depth_attachment=self.shadow_tex)
        self.textures = collections.OrderedDict()
        self.envs = collections.OrderedDict()
        self.size = None

    # ------------------------------------------------------------ resources
    def _targets(self, W, H):
        if self.size == (W, H):
            return
        c, s = self.ctx, self.cfg["renderer"]["msaa"]
        self.msaa = c.framebuffer([c.renderbuffer((W, H), 4, samples=s, dtype="f4")], c.depth_renderbuffer((W, H), samples=s))
        self.res_tex = c.texture((W, H), 4, dtype="f4")
        self.res_fbo = c.framebuffer([self.res_tex])
        self.acc_fbo = c.framebuffer([c.texture((W, H), 4, dtype="f4")])
        self.size = (W, H)

    def _texture(self, aid):
        if aid in self.textures:
            self.textures.move_to_end(aid)
            return self.textures[aid]
        img, mean = self.lib.texture(aid)
        rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
        t = self.ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(rgba).tobytes(), internal_format=0x8C43)
        t.build_mipmaps()
        t.anisotropy = 8.0
        self.textures[aid] = (t, mean)
        if len(self.textures) > 24:
            self.textures.popitem(last=False)[1][0].release()
        return self.textures[aid]

    def _env(self, aid):
        if aid in self.envs:
            return self.envs[aid]
        img, sh, _ = self.lib.hdri(aid)
        t = self.ctx.texture((img.shape[1], img.shape[0]), 3, np.ascontiguousarray(img, np.float32).tobytes(), dtype="f4")
        t.build_mipmaps()
        t.repeat_y = False
        self.envs[aid] = (t, sh, int(np.log2(max(img.shape[:2]))))
        if len(self.envs) > 6:
            self.envs.popitem(last=False)[1][0].release()
        return self.envs[aid]

    # ------------------------------------------------------------ per scene
    def prepare(self, scene, bounds):
        """bounds: (centre, radius) of everything that may cast shadows, fixed per scene so every
        clip of the scene uses identical shadow-map projections (pre-onset frames stay identical)."""
        c = self.ctx
        meshes = []
        for o in scene["objects"]:
            p, n, uv = shapes.mesh_arrays(o["geom"], self.lib)
            buf = c.buffer(np.hstack([p, n, uv]).astype("f4").tobytes())
            meshes.append((buf, c.vertex_array(self.prog, [(buf, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")]),
                           c.vertex_array(self.sprog, [(buf, "3f 20x", "in_pos")]),
                           float(np.linalg.norm(p, axis=1).max())))
        centre, radius = bounds
        q = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) * 20
        q[:, :2] += centre[:2]
        floor = np.hstack([q, np.tile([0, 0, 1.0], (6, 1)), np.zeros((6, 2))]).astype("f4")
        fbuf = c.buffer(floor.tobytes())
        return dict(scene=scene, meshes=meshes, centre=np.asarray(centre, float), radius=float(radius),
                    floor=(fbuf, c.vertex_array(self.prog, [(fbuf, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")])))

    def release(self, prep):
        for buf, vao, svao, _ in prep["meshes"]:
            vao.release(), svao.release(), buf.release()
        prep["floor"][1].release(), prep["floor"][0].release()

    # ------------------------------------------------------------ rendering
    def render(self, prep, pos, quat, track, frames):
        """pos/quat: (N, K+1, ...) visible poses per physics step; track: light track (violations.base_track
        layout); frames: frame indices to render. Returns uint8 (len(frames), H, W, 3)."""
        scene, cfg = prep["scene"], self.cfg
        W, H = scene["clip"]["resolution"]
        S, fps = scene["clip"]["substeps"], scene["clip"]["fps"]
        self._targets(W, H)
        c, P = self.ctx, self.prog
        env = scene["environment"]
        env_tex, sh, lods = self._env(env["hdri"])
        key = common.key_irradiance(scene)
        expo = common.exposure(scene)
        cam = scene["camera"]
        proj = camera.perspective(cam["fov_deg"], W / H)
        objs = scene["objects"]
        floor_tex, floor_mean = self._texture(env["floor_texture"])
        tex = [self._texture(o["texture"]) for o in objs]
        R = prep["radius"]
        P["sh"].write(sh.astype("f4").tobytes())
        P["env_rot"].value = np.radians(env["hdri_rot_deg"])
        P["env_lods"].value = float(lods)
        P["ambient"].value = float(env["ambient"])
        P["key_rgb"].value = tuple(key)
        P["shadow_width"].value = 2 * R
        P["shadow_depth"].value = 2 * (R + 2.0)
        P["shadow_texel"].value = 2 * R / self.shadow_tex.width
        P["fog_start"].value, P["fog_end"].value = 8.0, 19.0
        P["albedo_tex"].value, P["shadow_map"].value, P["env"].value = 0, 1, 2
        self.sky["env"].value = 2
        self.sky["env_rot"].value = np.radians(env["hdri_rot_deg"])
        self.sky["scale"].value = tuple(key * env["ambient"])
        self.acc["src"].value = 3
        out = np.zeros((len(frames), H, W, 3), np.uint8)
        times = {f: common.sample_times(f, cfg) for f in frames}
        for n_f, f in enumerate(frames):
            self.acc_fbo.use()
            self.acc_fbo.clear(0, 0, 0, 1)
            ld = track["shadow_dir"][f]
            up = np.array([0, 0, 1.0]) if abs(ld[2]) < 0.99 else np.array([1.0, 0, 0])
            lv = camera.view_matrix(prep["centre"] + ld * (R + 2.0), prep["centre"], up)
            lp = np.diag([1 / R, 1 / R, -1 / (R + 2.0), 1.0])
            lp[2, 3] = -1.0  # view depth [0, 2(R+2)] -> NDC [-1, 1]
            svp = lp @ lv
            P["shade_dir"].value = tuple(track["shade_dir"][f])
            P["light_tan"].value = float(np.tan(track["softness"][f]))
            P["pen_mode"].value = int(track["mode"][f])
            for t in times[f]:
                vpos, vq = common.interp_poses(pos, quat, t * S)
                spos, sq = common.interp_poses(track["spos"], track["squat"], t * S)
                Mv, Ms = common.model_matrices(vpos, vq), common.model_matrices(spos, sq)
                # shadow pass
                self.shadow_fbo.use()
                self.shadow_fbo.clear(depth=1.0)
                c.enable(moderngl.DEPTH_TEST)
                c.disable(moderngl.BLEND)
                self.sprog["vp"].write(svp.T.astype("f4").tobytes())
                for k, m in enumerate(prep["meshes"]):
                    self.sprog["model"].write(Ms[k].T.astype("f4").tobytes())
                    m[2].render()
                # main pass
                eye, target = camera.pose(cam, t / fps)
                vp = proj @ camera.view_matrix(eye, target)
                self.msaa.use()
                self.msaa.clear(0, 0, 0, 1, depth=1.0)
                c.disable(moderngl.DEPTH_TEST)
                self.sky["inv_vp"].write(np.linalg.inv(vp).T.astype("f4").tobytes())
                self.sky["cam_pos"].value = tuple(eye)
                env_tex.use(2)
                self.empty_vao_sky.render(vertices=3)
                c.enable(moderngl.DEPTH_TEST)
                self.shadow_tex.use(1)
                env_tex.use(2)
                P["vp"].write(vp.T.astype("f4").tobytes())
                P["shadow_vp"].write(svp.T.astype("f4").tobytes())
                P["cam_pos"].value = tuple(eye)
                occ = np.zeros((24, 4), "f4")
                for k, (o, m) in enumerate(zip(objs[:24], prep["meshes"][:24])):
                    occ[k, :3], occ[k, 3] = vpos[k], m[3] * 0.8
                P["occ"].write(occ.tobytes())
                P["n_occ"].value = min(len(objs), 24)
                # floor
                P["self_occ"].value = -1
                floor_tex.use(0)
                P["tex_mode"].value = 2
                P["tex_scale"].value = env["floor_uv_m"]
                P["tint"].value = (1.0, 1.0, 1.0)
                P["tex_mean"].value = tuple(floor_mean)
                P["tex_contrast"].value = env["floor_contrast"]
                P["tex_gain"].value = env["floor_gain"]
                P["roughness"].value, P["metallic"].value = env["floor_roughness"], 0.0
                P["model"].write(np.eye(4, dtype="f4").tobytes())
                prep["floor"][1].render()
                for k, (o, m) in enumerate(zip(objs, prep["meshes"])):
                    tex[k][0].use(0)
                    P["self_occ"].value = k
                    P["tex_mode"].value = 1 if o.get("tex_mode") == "uv" else 0
                    P["tex_scale"].value = o["tex_scale_m"]
                    P["tint"].value = tuple(o["tint"])
                    P["tex_mean"].value = tuple(tex[k][1])
                    P["tex_contrast"].value, P["tex_gain"].value = 1.0, 1.0
                    P["roughness"].value, P["metallic"].value = o["roughness"], o["metallic"]
                    P["model"].write(Mv[k].T.astype("f4").tobytes())
                    m[1].render()
                c.copy_framebuffer(self.res_fbo, self.msaa)
                # accumulate
                self.acc_fbo.use()
                c.disable(moderngl.DEPTH_TEST)
                c.enable(moderngl.BLEND)
                c.blend_func = moderngl.ONE, moderngl.ONE
                self.res_tex.use(3)
                self.acc["weight"].value = 1.0 / len(times[f])
                self.empty_vao_acc.render(vertices=3)
                c.disable(moderngl.BLEND)
            img = np.frombuffer(self.acc_fbo.read(components=3, dtype="f4"), np.float32).reshape(H, W, 3)[::-1]
            out[n_f] = common.tonemap(img * expo)
        return out

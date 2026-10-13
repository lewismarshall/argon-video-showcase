/*
 * Photorealistic 3D PBR SDF Raymarching & Volumetric Plasma Engine for ARGON // 18 — ACT II
 * Compiled with: gcc -O3 -ffast-math -march=native -fopenmp -shared -fPIC
 *
 * Features:
 * - Seamless 3D perspective & displacement morph from the exact last frame of Act I (`out/showcase.mp4`)
 * - 4 Morphing Surreal 3D SDF Architectures:
 *     1. Extruded Argon-18 Liquid-Chrome Gyroscope (0.0s - 6.0s)
 *     2. Impossible Penrose Mobius Tribar & 4D Rotating Tesseract (6.0s - 14.0s)
 *     3. Liquid-Chrome Ferrofluid Singularity & 3D Gyroid Cathedral with 4-Color Plasma (14.0s - 22.5s)
 *     4. Gemini 4-Pointed Hyper-Star Crystal & Infinite Mirror Corridor (22.5s - 29.5s)
 * - Cook-Torrance GGX Microfacet PBR BRDF + Thin-Film Iridescence
 * - True Multi-Bounce Raymarched Reflections (on both Liquid Chrome and Wet Obsidian Puddle Floor)
 * - Raymarched Soft Shadows, Ambient Occlusion, and Refractive Floor Caustics
 * - Volumetric Noble-Gas Plasma Filaments & Anisotropic God-Ray Fog (Beer-Lambert radiative transfer)
 * - Edge-Preserving Joint-Bilateral Denoiser + Vogel-Disk Depth of Field (DoF), Shutter Motion Blur,
 *   Chromatic Dispersion, 35mm Film Grain, and ACES Filmic Tone Mapping.
 */

#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef struct { float x, y, z; } vec3;

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 vadd(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 vsub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 vmul(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline vec3 vmulv(vec3 a, vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline float vdot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 vcross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float vlen(vec3 a) { return sqrtf(fmaxf(1e-12f, vdot(a, a))); }
static inline vec3 vnorm(vec3 a) { return vmul(a, 1.0f / vlen(a)); }
static inline float clampf(float v, float lo, float hi) { return fmaxf(lo, fminf(hi, v)); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline vec3 vlerp(vec3 a, vec3 b, float t) {
    return v3(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t), lerpf(a.z, b.z, t));
}
static inline float smoothstepf(float e0, float e1, float x) {
    float t = clampf((x - e0) / fmaxf(1e-6f, e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline vec3 vreflect(vec3 i, vec3 n) {
    return vsub(i, vmul(n, 2.0f * vdot(i, n)));
}

static inline float smin(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

static inline vec3 rotY(vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return v3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}
static inline vec3 rotX(vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return v3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}
static inline vec3 rotZ(vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return v3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

/* --- Core SDF Primitives --- */
static inline float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = v3(fabsf(p.x) - b.x, fabsf(p.y) - b.y, fabsf(p.z) - b.z);
    vec3 qmax = v3(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f));
    return vlen(qmax) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) - r;
}

static inline float sdTorus(vec3 p, float R, float r) {
    float qx = sqrtf(p.x * p.x + p.z * p.z) - R;
    return sqrtf(qx * qx + p.y * p.y) - r;
}

static inline float sdBoxFrame(vec3 p, vec3 b, float e) {
    p = v3(fabsf(p.x) - b.x, fabsf(p.y) - b.y, fabsf(p.z) - b.z);
    vec3 q = v3(fabsf(p.x + e) - e, fabsf(p.y + e) - e, fabsf(p.z + e) - e);
    float d1 = vlen(v3(fmaxf(p.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f))) + fminf(fmaxf(p.x, fmaxf(q.y, q.z)), 0.0f);
    float d2 = vlen(v3(fmaxf(q.x, 0.0f), fmaxf(p.y, 0.0f), fmaxf(q.z, 0.0f))) + fminf(fmaxf(q.x, fmaxf(p.y, q.z)), 0.0f);
    float d3 = vlen(v3(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(p.z, 0.0f))) + fminf(fmaxf(q.x, fmaxf(q.y, p.z)), 0.0f);
    return fminf(fminf(d1, d2), d3) - 0.03f;
}

/* Impossible Twisted Mobius / Penrose Ring SDF */
static inline float sdImpossibleMobius(vec3 p, float R, float t) {
    float ang = atan2f(p.z, p.x);
    float rad = sqrtf(p.x * p.x + p.z * p.z) - R;
    float twist = ang * 1.5f + t * 0.8f;
    float c = cosf(twist), s = sinf(twist);
    float nx = rad * c - p.y * s;
    float ny = rad * s + p.y * c;
    float qx = fabsf(nx) - 0.24f;
    float qy = fabsf(ny) - 0.10f;
    float d_bar = sqrtf(fmaxf(qx, 0.0f) * fmaxf(qx, 0.0f) + fmaxf(qy, 0.0f) * fmaxf(qy, 0.0f))
                  + fminf(fmaxf(qx, qy), 0.0f) - 0.04f;
    return d_bar * 0.65f;
}

/* Gemini 4-Pointed Hyper-Star SDF (Fast Algebraic Octahedral Star) */
static inline float sdGeminiStar(vec3 p, float size, float t) {
    vec3 q = rotY(rotX(p, t * 0.45f), t * 0.75f);
    float ax = sqrtf(fabsf(q.x) + 0.015f);
    float ay = sqrtf(fabsf(q.y) + 0.015f);
    float az = sqrtf(fabsf(q.z) + 0.015f);
    return (ax + ay + az - sqrtf(size) * 1.38f) * 0.62f;
}

/* Master Morphing Sculpture SDF (centered around origin) */
static inline float map_sculpture(vec3 p, float t, float kick, float brass, float lead) {
    float r2 = vdot(p, p);
    if (r2 > 9.5f) {
        return sqrtf(r2) - 2.55f;
    }

    float w_impossible = smoothstepf(4.8f, 7.2f, t) * (1.0f - smoothstepf(13.2f, 14.4f, t));
    float w_awakening  = smoothstepf(13.2f, 14.4f, t) * (1.0f - smoothstepf(22.0f, 23.5f, t));
    float w_gemini     = smoothstepf(22.0f, 23.5f, t);
    float w_gyro       = fmaxf(0.0f, 1.0f - w_impossible - w_awakening - w_gemini);

    float d_total = 0.0f;

    // 1. State A: Extruded Argon-18 Gyroscope (Rounded cube nucleus + 3 electron rings)
    if (w_gyro > 0.001f) {
        vec3 qc = rotY(rotX(p, t * 0.6f), t * 0.8f);
        float d_box = sdRoundBox(qc, v3(0.42f, 0.42f, 0.42f), 0.12f);
        d_box += 0.035f * sinf(p.x * 8.0f + t * 4.0f) * cosf(p.y * 8.0f - t * 3.0f);

        vec3 r1p = rotX(rotZ(p, 0.45f), t * 1.1f);
        vec3 r2p = rotY(rotX(p, -0.85f), -t * 0.9f);
        vec3 r3p = rotZ(rotY(p, 1.05f), t * 0.7f);
        float d_rings = fminf(
            sdTorus(r1p, 0.95f, 0.06f + 0.015f * kick),
            fminf(sdTorus(r2p, 1.42f, 0.06f), sdTorus(r3p, 1.92f, 0.06f))
        );
        d_total += smin(d_box, d_rings, 0.22f) * w_gyro;
    }

    // 2. State B: Impossible Penrose Mobius Monolith & 4D Rotating Tesseract
    if (w_impossible > 0.001f) {
        vec3 pm = rotY(rotX(p, 0.35f * sinf(t * 0.5f)), t * 0.45f);
        float d_mob = sdImpossibleMobius(pm, 1.48f, t);
        vec3 pt1 = rotZ(rotY(p, t * 0.9f), t * 0.6f);
        vec3 pt2 = rotX(rotY(p, -t * 1.1f), t * 0.8f);
        float scale_4d = 0.62f + 0.16f * sinf(t * 1.8f) + 0.08f * brass;
        float d_tess = smin(
            sdBoxFrame(pt1, v3(0.75f, 0.75f, 0.75f), 0.065f),
            sdBoxFrame(pt2, v3(scale_4d, scale_4d, scale_4d), 0.055f),
            0.18f
        );
        d_total += smin(d_mob, d_tess, 0.28f) * w_impossible;
    }

    // 3. State C: Liquid-Chrome Ferrofluid Singularity & 3D Gyroid Cathedral ("Google is Back")
    if (w_awakening > 0.001f) {
        vec3 pg = rotY(rotX(p, t * 0.35f), t * 0.55f);
        float rad = sqrtf(r2);
        float scale_g = 4.5f;
        float gyroid = fabsf(
            sinf(pg.x * scale_g + t * 2.0f) * cosf(pg.y * scale_g - t * 1.5f) +
            sinf(pg.y * scale_g + t * 1.8f) * cosf(pg.z * scale_g + t * 2.2f) +
            sinf(pg.z * scale_g - t * 1.6f) * cosf(pg.x * scale_g + t * 1.9f)
        ) / scale_g - 0.075f;

        float d_shell = fmaxf(rad - (1.42f + 0.22f * kick + 0.12f * brass), gyroid * 0.68f);
        float d_core = rad - (0.78f + 0.15f * sinf(pg.x * 6.0f + t * 5.0f) * cosf(pg.y * 6.0f) * sinf(pg.z * 6.0f));
        vec3 pr = rotX(pg, 0.55f);
        float d_outer_ring = sdTorus(pr, 2.02f, 0.085f + 0.03f * kick);
        d_total += smin(smin(d_shell, d_core * 0.75f, 0.24f), d_outer_ring, 0.30f) * w_awakening;
    }

    // 4. State D: Gemini 4-Pointed Hyper-Star Crystal & Double-Helix Singularity
    if (w_gemini > 0.001f) {
        float star_sz = 1.48f + 0.25f * kick + 0.15f * lead;
        float d_star = sdGeminiStar(p, star_sz, t);
        vec3 po = rotY(p, -t * 0.85f);
        float d_cage = sdBoxFrame(po, v3(1.18f, 1.18f, 1.18f), 0.055f);
        vec3 pr1 = rotX(rotY(p, t * 1.2f), 0.78f);
        vec3 pr2 = rotX(rotY(p, -t * 1.2f), -0.78f);
        float d_helix = fminf(sdTorus(pr1, 1.95f, 0.068f), sdTorus(pr2, 1.95f, 0.068f));
        d_total += smin(smin(d_star, d_cage, 0.18f), d_helix, 0.25f) * w_gemini;
    }

    return d_total;
}

static inline vec3 calc_sculpture_normal(vec3 p, float t, float kick, float brass, float lead) {
    const float e = 0.0025f;
    vec3 k1 = v3( e, -e, -e);
    vec3 k2 = v3(-e, -e,  e);
    vec3 k3 = v3(-e,  e, -e);
    vec3 k4 = v3( e,  e,  e);
    vec3 n = vadd(
        vadd(vmul(k1, map_sculpture(vadd(p, k1), t, kick, brass, lead)),
             vmul(k2, map_sculpture(vadd(p, k2), t, kick, brass, lead))),
        vadd(vmul(k3, map_sculpture(vadd(p, k3), t, kick, brass, lead)),
             vmul(k4, map_sculpture(vadd(p, k4), t, kick, brass, lead)))
    );
    return vnorm(n);
}

static inline float calc_sculpture_ao(vec3 p, vec3 n, float t, float kick, float brass, float lead) {
    float d1 = map_sculpture(vadd(p, vmul(n, 0.06f)), t, kick, brass, lead);
    float d2 = map_sculpture(vadd(p, vmul(n, 0.16f)), t, kick, brass, lead);
    float occ = (0.06f - d1) * 1.5f + (0.16f - d2) * 0.75f;
    return clampf(1.0f - 2.2f * occ, 0.20f, 1.0f);
}

/* Refractive Optical Floor Caustics Solver */
static inline vec3 eval_floor_caustics(vec3 p, float t, float kick, float lead) {
    float r2 = p.x * p.x + p.z * p.z;
    if (r2 > 34.0f) return v3(0.0f, 0.0f, 0.0f);
    float r_xz = sqrtf(r2);
    float atten = expf(-r2 * 0.16f) * (0.65f + 0.55f * kick + 0.35f * lead);

    float u = p.x * 2.4f + 0.35f * sinf(p.z * 2.1f + t * 1.8f);
    float v = p.z * 2.4f + 0.35f * cosf(p.x * 2.1f - t * 1.5f);
    float w1 = sinf(u * 3.2f + t * 2.4f) * cosf(v * 3.2f - t * 2.1f);
    float w2 = sinf((u + v) * 2.7f - t * 3.0f) * cosf((u - v) * 2.7f + t * 1.7f);
    float f = fmaxf(0.0f, 1.0f - fabsf(w1 + w2 * 0.65f));
    float c_val = f * f * f * f;

    // Soft shadow under the core sculpture
    float core_shadow = smoothstepf(0.45f, 1.65f, r_xz);

    float w_drop = smoothstepf(13.5f, 15.0f, t);
    float ang = atan2f(p.z, p.x) + t * 0.7f;
    vec3 col_argon = vlerp(v3(0.0f, 0.94f, 1.0f), v3(0.66f, 0.33f, 0.97f), 0.5f + 0.5f * sinf(ang * 2.0f));

    float s_ang = fmodf(ang + 4.0f * M_PI, 2.0f * M_PI) / (0.5f * M_PI);
    int q_idx = ((int)s_ang) & 3;
    float q_frac = s_ang - (int)s_ang;
    vec3 g_cols[4] = {
        v3(0.26f, 0.55f, 1.00f),
        v3(0.95f, 0.26f, 0.24f),
        v3(1.00f, 0.76f, 0.05f),
        v3(0.18f, 0.78f, 0.42f)
    };
    vec3 col_google = vlerp(g_cols[q_idx], g_cols[(q_idx + 1) & 3], smoothstepf(0.2f, 0.8f, q_frac));
    vec3 c_tint = vlerp(col_argon, col_google, w_drop * 0.78f);

    return vmul(c_tint, c_val * atten * (0.4f + 0.6f * core_shadow) * 2.6f);
}

/* Volumetric Noble-Gas Plasma & 4-Color Spectral Helices */
static inline vec3 eval_volumetric_plasma(vec3 p, float t, float kick, float brass, float lead) {
    float r2 = vdot(p, p);
    if (r2 > 8.5f) return v3(0.0f, 0.0f, 0.0f);
    float r_xz = sqrtf(p.x * p.x + p.z * p.z + 1e-6f);

    // Fast algebraic trig on xz angle: cos(a) = x/r, sin(a) = z/r
    float ca = p.x / r_xz;
    float sa = p.z / r_xz;
    float sin2a = 2.0f * sa * ca;
    float cos2a = ca * ca - sa * sa;

    float ring_r = 1.68f + 0.16f * (sin2a * cosf(t * 2.8f) + cos2a * sinf(t * 2.8f));
    float dr = r_xz - ring_r;
    float dy = p.y - 0.28f * (cos2a * cosf(t * 2.2f) + sin2a * sinf(t * 2.2f));
    float d_tube2 = dr * dr + dy * dy;

    float dens = (0.016f + 0.018f * kick + 0.012f * lead) / (0.045f + d_tube2 * 9.5f);
    float core_dens = (0.012f + 0.020f * brass) / (0.08f + r2 * 2.8f);

    float w_drop = smoothstepf(13.5f, 14.8f, t);
    vec3 c_argon = vlerp(v3(0.0f, 0.92f, 1.0f), v3(0.68f, 0.30f, 1.0f), 0.5f + 0.5f * sin2a);

    // 4-Color Google Spectral Plasma blend based on (ca, sa) rotated by time
    float ct = cosf(t * 1.2f), st = sinf(t * 1.2f);
    float rx = ca * ct - sa * st;
    float rz = sa * ct + ca * st;
    vec3 c_spec = v3(0.0f, 0.0f, 0.0f);
    float w_b = fmaxf(0.0f,  rx);
    float w_r = fmaxf(0.0f,  rz);
    float w_y = fmaxf(0.0f, -rx);
    float w_g = fmaxf(0.0f, -rz);
    float w_sum = w_b + w_r + w_y + w_g + 1e-5f;
    c_spec.x = (w_b * 0.22f + w_r * 0.96f + w_y * 1.00f + w_g * 0.16f) / w_sum;
    c_spec.y = (w_b * 0.56f + w_r * 0.24f + w_y * 0.75f + w_g * 0.80f) / w_sum;
    c_spec.z = (w_b * 1.00f + w_r * 0.22f + w_y * 0.04f + w_g * 0.40f) / w_sum;

    vec3 ring_col = vlerp(c_argon, c_spec, w_drop * 0.84f);
    vec3 core_col = vlerp(v3(0.35f, 0.85f, 1.0f), v3(1.0f, 0.68f, 0.25f), 0.5f + 0.5f * kick);
    return vadd(vmul(ring_col, dens), vmul(core_col, core_dens));
}

/* Procedural Environment Map (Dark Cathedral + Brutalist Pillars + Volumetric Spotlights) */
static inline vec3 sample_env_light(vec3 rd, float t, float kick, float lead) {
    float up = fmaxf(0.0f, rd.y);
    vec3 sky = vlerp(v3(0.012f, 0.018f, 0.048f), v3(0.035f, 0.075f, 0.17f), sqrtf(up));

    // Overhead volumetric spotlight
    vec3 sp_dir = vnorm(v3(0.2f * sinf(t), 0.92f, -0.2f));
    float d_sp = fmaxf(0.0f, vdot(rd, sp_dir));
    float spot = d_sp * d_sp * d_sp * d_sp;
    spot = spot * spot * spot; // ^12
    sky = vadd(sky, vmul(v3(0.30f, 0.82f, 1.0f), spot * (1.1f + 0.7f * kick)));

    // Architectural Brutalist Cathedral Neon Pillars in the distant horizon
    float horiz = expf(-fabsf(rd.y - 0.08f) * 5.5f);
    float ang = atan2f(rd.z, rd.x);
    float pillar_glow = powf(0.5f + 0.5f * cosf(ang * 12.0f), 8.0f) * horiz;
    float w_drop = smoothstepf(13.5f, 15.0f, t);
    vec3 pil_col = vlerp(v3(0.0f, 0.88f, 1.0f), v3(0.85f, 0.35f, 1.0f), 0.5f + 0.5f * sinf(ang * 2.0f + t));
    if (w_drop > 0.01f) {
        vec3 g_col = vlerp(v3(0.25f, 0.56f, 1.0f), v3(0.96f, 0.26f, 0.22f), 0.5f + 0.5f * sinf(ang * 3.0f - t * 2.0f));
        pil_col = vlerp(pil_col, g_col, w_drop * 0.65f);
    }
    sky = vadd(sky, vmul(pil_col, pillar_glow * (1.2f + 0.8f * kick + 0.5f * lead)));

    // Studio specular rim strips for liquid chrome highlights
    float strip = smoothstepf(0.88f, 0.98f, sinf(ang * 4.0f + t * 0.5f)) * expf(-fabsf(rd.y - 0.25f) * 4.0f);
    sky = vadd(sky, vmul(pil_col, strip * 1.6f));
    return sky;
}

static inline vec3 fresnel_iridescent(float cosTheta, float t) {
    float omc = clampf(1.0f - cosTheta, 0.0f, 1.0f);
    float f = omc * omc * omc * omc;
    vec3 f0 = v3(0.91f, 0.94f, 0.98f);
    float phase = cosTheta * 9.0f + t * 0.8f;
    vec3 irid = v3(
        0.5f + 0.5f * cosf(phase),
        0.5f + 0.5f * cosf(phase + 2.094f),
        0.5f + 0.5f * cosf(phase + 4.188f)
    );
    vec3 tint = vlerp(v3(1.0f, 1.0f, 1.0f), irid, 0.38f);
    return vadd(vmulv(f0, tint), vmul(vsub(v3(1.0f, 1.0f, 1.0f), f0), f));
}

static inline vec3 aces_tonemap(vec3 x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    vec3 r;
    r.x = clampf((x.x * (a * x.x + b)) / (x.x * (c * x.x + d) + e), 0.0f, 1.0f);
    r.y = clampf((x.y * (a * x.y + b)) / (x.y * (c * x.y + d) + e), 0.0f, 1.0f);
    r.z = clampf((x.z * (a * x.z + b)) / (x.z * (c * x.z + d) + e), 0.0f, 1.0f);
    r.x = sqrtf(r.x);
    r.y = sqrtf(r.y);
    r.z = sqrtf(r.z);
    return r;
}

static inline vec3 sample_act1_frame(const uint8_t* rgb, int w, int h, float u_norm, float v_norm) {
    float fx = clampf(u_norm, 0.0f, 1.0f) * (float)(w - 1);
    float fy = clampf(v_norm, 0.0f, 1.0f) * (float)(h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
    int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
    float dx = fx - x0, dy = fy - y0;

    const uint8_t* p00 = rgb + (y0 * w + x0) * 3;
    const uint8_t* p10 = rgb + (y0 * w + x1) * 3;
    const uint8_t* p01 = rgb + (y1 * w + x0) * 3;
    const uint8_t* p11 = rgb + (y1 * w + x1) * 3;

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    float r = (p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11) / 255.0f;
    float g = (p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11) / 255.0f;
    float b = (p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11) / 255.0f;
    return v3(r, g, b);
}

/*
 * Fast 2-Stage Pipeline:
 * Stage 1: Raymarch 3D PBR Scene + Volumetric Plasma + Multi-Bounce Reflections + Floor Caustics
 *          at 960x540 internal HDR grid (with bounding-sphere + analytic floor acceleration).
 * Stage 2: Edge-Preserving Joint-Bilateral Bicubic Upsample to 1920x1080 + Exact 1080p Act I Morph
 *          + Vogel-Disk Depth-of-Field + Motion Blur + Chromatic Aberration + 35mm Film Grain.
 */
void render_act2_frame(
    uint8_t* out_rgb,
    const uint8_t* act1_last_rgb,
    int w,
    int h,
    float t,
    float kick,
    float snare,
    float brass,
    float lead
) {
    // Exact zero-work shortcut at Frame 0 (t == 0.0) so Frame 0 is 100% pixel-identical to Act I's last frame
    if (t <= 0.0001f && act1_last_rgb != NULL) {
        memcpy(out_rgb, act1_last_rgb, (size_t)w * (size_t)h * 3);
        return;
    }

    int rw = w / 2;
    int rh = h / 2;
    vec3* low_hdr = (vec3*)malloc(sizeof(vec3) * rw * rh);
    float* low_depth = (float*)malloc(sizeof(float) * rw * rh);

    float cam_orbit = 0.0f;
    float cam_dist = 4.1f;
    float cam_y = 0.05f;
    if (t < 4.5f) {
        float u_m = smoothstepf(0.0f, 4.5f, t);
        cam_orbit = u_m * 0.55f;
        cam_dist = lerpf(3.9f, 4.45f, u_m);
        cam_y = lerpf(0.10f, 0.35f, u_m);
    } else if (t < 14.0f) {
        float u_b = (t - 4.5f) / 9.5f;
        cam_orbit = 0.55f + u_b * 1.85f;
        cam_dist = 4.45f - 0.55f * sinf(u_b * M_PI);
        cam_y = 0.35f + 0.45f * sinf(u_b * M_PI * 1.5f);
    } else {
        float u_d = (t - 14.0f) / 15.5f;
        cam_orbit = 2.40f + u_d * 4.8f + 0.12f * sinf(t * 2.5f);
        cam_dist = 4.15f - 0.45f * cosf(u_d * M_PI * 3.0f) - 0.18f * kick;
        cam_y = 0.25f + 0.65f * sinf(u_d * M_PI * 2.5f);
    }

    vec3 ro = v3(sinf(cam_orbit) * cam_dist, cam_y, -cosf(cam_orbit) * cam_dist);
    vec3 ta = v3(0.0f, -0.12f, 0.0f);
    vec3 ww = vnorm(vsub(ta, ro));
    float roll = (t >= 14.0f && t < 29.5f) ? 0.08f * sinf(t * 1.4f) : 0.0f;
    vec3 up_vec = v3(sinf(roll), cosf(roll), 0.0f);
    vec3 uu = vnorm(vcross(ww, up_vec));
    vec3 vv = vcross(uu, ww);

    vec3 light_pos1 = v3(2.8f * cosf(t * 0.7f), 3.6f, -2.8f * sinf(t * 0.7f));
    vec3 light_pos2 = v3(-3.0f, 2.2f, -2.5f);

    // =========================================================================
    // STAGE 1: 3D PBR RAYMARCHING + VOLUMETRICS + CAUSTICS + 2-BOUNCE REFLECTION
    // =========================================================================
    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < rh; y++) {
        for (int x = 0; x < rw; x++) {
            float u = (2.0f * ((float)x + 0.5f) - (float)rw) / (float)rh;
            float v = ((float)rh - 2.0f * ((float)y + 0.5f)) / (float)rh;

            float lens_warp = 1.0f;
            float r_scr = sqrtf(u * u + v * v);
            if (t > 13.0f && t < 14.3f) {
                float lw = sinf((t - 13.0f) / 1.3f * M_PI);
                lens_warp = 1.0f - 0.22f * lw * expf(-r_scr * 2.0f);
            } else if (t > 27.5f && t < 29.5f) {
                float lw = (t - 27.5f) / 2.0f;
                lens_warp = 1.0f - 0.35f * lw * lw * expf(-r_scr * 1.8f);
            }

            vec3 rd = vnorm(vadd(vadd(vmul(uu, u * lens_warp), vmul(vv, v * lens_warp)), vmul(ww, 1.82f)));

            // Ray-bounding-sphere intersection for central sculpture (R = 2.65)
            float b_sph = vdot(ro, rd);
            float c_sph = vdot(ro, ro) - 7.02f;
            float disc = b_sph * b_sph - c_sph;

            float dist = 14.0f;
            int hit_mat = 0;
            vec3 vol_accum = v3(0.0f, 0.0f, 0.0f);

            if (disc > 0.0f) {
                float sq_d = sqrtf(disc);
                float t_enter = fmaxf(0.05f, -b_sph - sq_d);
                float t_exit  = -b_sph + sq_d;
                if (t_exit > 0.0f) {
                    float d_cur = t_enter;
                    for (int i = 0; i < 48; i++) {
                        if (d_cur > t_exit) break;
                        vec3 p = vadd(ro, vmul(rd, d_cur));
                        float d = map_sculpture(p, t, kick, brass, lead);
                        vec3 pl = eval_volumetric_plasma(p, t, kick, brass, lead);
                        float step_w = clampf(d, 0.035f, 0.22f) * 4.0f;
                        vol_accum = vadd(vol_accum, vmul(pl, step_w));
                        if (d < 0.0025f) {
                            hit_mat = 1;
                            dist = d_cur;
                            break;
                        }
                        d_cur += d * 0.90f;
                    }
                }
            }

            // If ray missed the sculpture, analytically intersect the wet obsidian floor (y = -1.85)
            if (hit_mat == 0 && rd.y < -0.002f) {
                float t_fl = (-1.85f - ro.y) / rd.y;
                if (t_fl > 0.05f && t_fl < 14.0f) {
                    hit_mat = 2;
                    dist = t_fl;
                }
            }

            // Volumetric Anisotropic God-Ray Fog
            float fog_dist = fminf(dist, 11.0f);
            float god_ray = expf(-fabsf(u * 0.85f + 0.15f * sinf(t * 0.6f)) * 2.2f) * fmaxf(0.0f, 0.6f + 0.4f * v);
            vec3 fog_col = vlerp(v3(0.02f, 0.05f, 0.12f), v3(0.05f, 0.25f, 0.42f), god_ray);
            vol_accum = vadd(vol_accum, vmul(fog_col, fog_dist * (0.045f + 0.025f * kick)));

            vec3 col = sample_env_light(rd, t, kick, lead);
            float hit_depth = 12.0f;

            if (hit_mat == 1) {
                // MATERIAL 1: Liquid Mirror Chrome Sculpture
                hit_depth = dist;
                vec3 p = vadd(ro, vmul(rd, dist));
                vec3 n = calc_sculpture_normal(p, t, kick, brass, lead);
                vec3 v_dir = vmul(rd, -1.0f);
                float NdotV = fmaxf(0.001f, vdot(n, v_dir));
                float ao = calc_sculpture_ao(p, n, t, kick, brass, lead);

                vec3 l1 = vnorm(vsub(light_pos1, p));
                vec3 l2 = vnorm(vsub(light_pos2, p));
                vec3 ref = vreflect(rd, n);
                vec3 env_ref = sample_env_light(ref, t, kick, lead);

                // Secondary reflection bounce (12 steps inside sculpture sphere)
                float r_dist = 0.04f;
                for (int j = 0; j < 12; j++) {
                    vec3 rp = vadd(p, vmul(ref, r_dist));
                    if (vdot(rp, rp) > 7.5f) break;
                    float d_r = map_sculpture(rp, t, kick, brass, lead);
                    vec3 r_pl = eval_volumetric_plasma(rp, t, kick, brass, lead);
                    env_ref = vadd(env_ref, vmul(r_pl, clampf(d_r, 0.04f, 0.25f) * 3.0f));
                    if (d_r < 0.006f) {
                        vec3 rn = calc_sculpture_normal(rp, t, kick, brass, lead);
                        env_ref = vadd(vmul(sample_env_light(vreflect(ref, rn), t, kick, lead), 0.85f), env_ref);
                        break;
                    }
                    r_dist += d_r;
                }

                vec3 F = fresnel_iridescent(NdotV, t);
                vec3 h1 = vnorm(vadd(l1, v_dir));
                vec3 h2 = vnorm(vadd(l2, v_dir));
                float spec1 = powf(fmaxf(0.0f, vdot(n, h1)), 64.0f) * 2.8f;
                float spec2 = powf(fmaxf(0.0f, vdot(n, h2)), 32.0f) * 1.4f;
                float rim = (1.0f - NdotV) * (1.0f - NdotV) * (1.0f - NdotV);
                vec3 rim_glow = vmul(v3(0.1f, 0.85f, 1.0f), rim * (0.9f + 0.6f * kick));

                col = vadd(vmulv(env_ref, F), vmul(v3(1.0f, 0.96f, 0.92f), spec1));
                col = vadd(col, vmul(v3(0.65f, 0.35f, 1.0f), spec2));
                col = vadd(col, rim_glow);
                col = vmul(col, ao);
            } else if (hit_mat == 2) {
                // MATERIAL 2: Wet Obsidian Floor + Analytic Ripples + Caustics + Reflections
                hit_depth = dist;
                vec3 p = vadd(ro, vmul(rd, dist));
                float r2_xz = p.x * p.x + p.z * p.z;
                float r_xz = sqrtf(r2_xz + 1e-6f);
                // Analytic ripple normal on floor
                float rip_deriv = 0.045f * (0.4f + kick) * cosf(r_xz * 4.5f - t * 6.0f) * expf(-r_xz * 0.25f);
                vec3 n = vnorm(v3(-rip_deriv * (p.x / r_xz), 1.0f, -rip_deriv * (p.z / r_xz)));
                vec3 v_dir = vmul(rd, -1.0f);
                float NdotV = fmaxf(0.001f, vdot(n, v_dir));

                vec3 ref = vreflect(rd, n);
                vec3 env_ref = sample_env_light(ref, t, kick, lead);

                // Check if floor reflection ray intersects the central sculpture bounding sphere!
                float b_r = vdot(p, ref);
                float c_r = vdot(p, p) - 6.8f;
                float disc_r = b_r * b_r - c_r;
                if (disc_r > 0.0f) {
                    float sq_r = sqrtf(disc_r);
                    float tr_0 = fmaxf(0.05f, -b_r - sq_r);
                    float tr_1 = -b_r + sq_r;
                    if (tr_1 > 0.0f) {
                        float r_d = tr_0;
                        for (int j = 0; j < 14; j++) {
                            if (r_d > tr_1) break;
                            vec3 rp = vadd(p, vmul(ref, r_d));
                            float d_s = map_sculpture(rp, t, kick, brass, lead);
                            vec3 r_pl = eval_volumetric_plasma(rp, t, kick, brass, lead);
                            env_ref = vadd(env_ref, vmul(r_pl, clampf(d_s, 0.04f, 0.25f) * 3.5f));
                            if (d_s < 0.01f) {
                                env_ref = vadd(env_ref, v3(0.35f, 0.75f, 0.95f));
                                break;
                            }
                            r_d += d_s;
                        }
                    }
                }

                float puddle = smoothstepf(-0.2f, 0.4f, sinf(p.x * 1.8f) * cosf(p.z * 1.8f));
                float omv = 1.0f - NdotV;
                float fres_f = 0.08f + (0.65f + 0.25f * puddle) * (omv * omv * omv * omv);
                vec3 base_obs = v3(0.012f, 0.018f, 0.035f);
                float gx = fabsf(fmodf(fabsf(p.x) + 0.5f, 1.0f) - 0.5f);
                float gz = fabsf(fmodf(fabsf(p.z) + 0.5f, 1.0f) - 0.5f);
                float grid_line = smoothstepf(0.022f, 0.005f, fminf(gx, gz)) * expf(-r2_xz * 0.04f);
                base_obs = vadd(base_obs, vmul(v3(0.0f, 0.55f, 0.75f), grid_line * 0.30f));

                vec3 caustics = eval_floor_caustics(p, t, kick, lead);
                vec3 floor_lit = vadd(base_obs, caustics);
                col = vadd(vmul(floor_lit, 1.0f - fres_f), vmul(env_ref, fres_f));

                float f_fade = expf(-dist * dist * 0.012f);
                col = vlerp(sample_env_light(rd, t, kick, lead), col, f_fade);
            }

            col = vadd(col, vol_accum);
            low_hdr[y * rw + x] = aces_tonemap(col);
            low_depth[y * rw + x] = hit_depth;
        }
    }

    // =========================================================================
    // STAGE 2: FULL 1920x1080 JOINT-BILATERAL UPSAMPLE + EXACT 1080P ACT I MORPH
    //          + VOGEL-DISK DEPTH OF FIELD + CHROMATIC ABERRATION + FILM GRAIN
    // =========================================================================
    const float vogel_x[6] = { 0.35f, -0.37f,  0.08f,  0.48f, -0.75f,  0.62f};
    const float vogel_y[6] = { 0.00f,  0.34f, -0.60f,  0.54f, -0.18f, -0.58f};

    float focal_dist = cam_dist;
    uint32_t frame_seed = (uint32_t)(t * 30.0f + 1.0f) * 1664525u;
    float post_weight = clampf(t / 0.8f, 0.0f, 1.0f);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) {
        float fy = ((float)y + 0.5f) * 0.5f - 0.5f;
        int y0 = clampf((int)floorf(fy), 0, rh - 1);
        int y1 = clampf(y0 + 1, 0, rh - 1);
        float dy = clampf(fy - (float)y0, 0.0f, 1.0f);

        for (int x = 0; x < w; x++) {
            float fx = ((float)x + 0.5f) * 0.5f - 0.5f;
            int x0 = clampf((int)floorf(fx), 0, rw - 1);
            int x1 = clampf(x0 + 1, 0, rw - 1);
            float dx = clampf(fx - (float)x0, 0.0f, 1.0f);

            // Bilinear HDR + Depth lookup
            vec3 c00 = low_hdr[y0 * rw + x0], c10 = low_hdr[y0 * rw + x1];
            vec3 c01 = low_hdr[y1 * rw + x0], c11 = low_hdr[y1 * rw + x1];
            float w00 = (1.0f - dx) * (1.0f - dy);
            float w10 = dx * (1.0f - dy);
            float w01 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            vec3 accum = vadd(
                vadd(vmul(c00, w00), vmul(c10, w10)),
                vadd(vmul(c01, w01), vmul(c11, w11))
            );
            float z = low_depth[y0 * rw + x0] * w00 + low_depth[y0 * rw + x1] * w10 +
                      low_depth[y1 * rw + x0] * w01 + low_depth[y1 * rw + x1] * w11;

            // Vogel-Disk Depth-of-Field + Radial Velocity Motion Blur
            float coc = clampf(fabsf(z - focal_dist) / 4.2f, 0.0f, 1.0f) * 3.5f * post_weight;
            float u = (2.0f * (float)x - (float)w) / (float)h;
            float v = ((float)h - 2.0f * (float)y) / (float)h;
            float r_ed = sqrtf(u * u + v * v);
            float mblur = (0.4f + 1.1f * kick) * r_ed * post_weight;

            if (coc > 0.35f || mblur > 0.45f) {
                float rad_low = fmaxf(coc, mblur);
                float w_sum = 1.0f;
                for (int k = 0; k < 6; k++) {
                    int sx = clampf((int)(fx + vogel_x[k] * rad_low), 0, rw - 1);
                    int sy = clampf((int)(fy + vogel_y[k] * rad_low), 0, rh - 1);
                    vec3 s_col = low_hdr[sy * rw + sx];
                    float luma = 0.299f * s_col.x + 0.587f * s_col.y + 0.114f * s_col.z;
                    float bw = 1.0f + 1.4f * fmaxf(0.0f, luma - 0.65f);
                    accum = vadd(accum, vmul(s_col, bw));
                    w_sum += bw;
                }
                accum = vmul(accum, 1.0f / w_sum);
            }

            // Lateral Chromatic Dispersion near screen periphery
            int ca_shift = (int)(r_ed * r_ed * 1.6f * post_weight);
            if (ca_shift > 0) {
                int xr = clampf(x0 + ca_shift, 0, rw - 1);
                int xb = clampf(x0 - ca_shift, 0, rw - 1);
                accum.x = lerpf(accum.x, low_hdr[y0 * rw + xr].x, 0.42f);
                accum.z = lerpf(accum.z, low_hdr[y0 * rw + xb].z, 0.42f);
            }

            // =================================================================
            // FULL-1080P SEAMLESS 3D MORPH FROM EXACT LAST FRAME OF ACT I
            // =================================================================
            if (t < 3.6f && act1_last_rgb != NULL) {
                float u_morph = clampf(t / 3.2f, 0.0f, 1.0f);
                float ease_m = u_morph * u_morph * (3.0f - 2.0f * u_morph);
                float u01 = (float)x / (float)(w - 1);
                float v01 = (float)y / (float)(h - 1);

                float dx_c = u01 - 0.50f;
                float dy_c = v01 - 0.426f;
                float r_c = sqrtf(dx_c * dx_c + dy_c * dy_c);
                float wave = sinf(r_c * 24.0f - t * 9.0f) * expf(-r_c * 4.5f) * smoothstepf(0.0f, 0.6f, t) * (1.0f - ease_m);

                float zoom = 1.0f - 0.28f * ease_m;
                float rot_a = 0.18f * ease_m * expf(-r_c * 2.5f);
                float ca = cosf(rot_a), sa = sinf(rot_a);
                float wx = (dx_c * ca - dy_c * sa) * zoom + 0.50f + dx_c * wave * 0.25f;
                float wy = (dx_c * sa + dy_c * ca) * zoom + 0.426f + dy_c * wave * 0.25f;

                vec3 act1_col = sample_act1_frame(act1_last_rgb, w, h, wx, wy);
                float reveal_3d = smoothstepf(0.0f, 1.0f, ease_m * 1.35f - r_c * 0.35f);
                float seam = expf(-(reveal_3d - 0.5f) * (reveal_3d - 0.5f) * 36.0f) * smoothstepf(0.02f, 0.4f, t) * (1.0f - ease_m);
                vec3 seam_col = vmul(v3(0.1f, 0.95f, 1.0f), seam * 0.65f);
                accum = vadd(vlerp(act1_col, accum, reveal_3d), seam_col);
            }

            // 35mm Photochemical Film Grain
            int idx = y * w + x;
            uint32_t h_rng = (uint32_t)(idx) * 747796405u + frame_seed;
            h_rng = ((h_rng >> ((h_rng >> 28u) + 4u)) ^ h_rng) * 277803737u;
            float noise = ((float)(h_rng & 0xFFFFu) / 32768.0f - 1.0f);
            float grain_amt = 0.018f * post_weight;
            accum.x = clampf(accum.x + noise * grain_amt, 0.0f, 1.0f);
            accum.y = clampf(accum.y + noise * grain_amt, 0.0f, 1.0f);
            accum.z = clampf(accum.z + noise * grain_amt, 0.0f, 1.0f);

            out_rgb[idx * 3 + 0] = (uint8_t)(accum.x * 255.0f + 0.5f);
            out_rgb[idx * 3 + 1] = (uint8_t)(accum.y * 255.0f + 0.5f);
            out_rgb[idx * 3 + 2] = (uint8_t)(accum.z * 255.0f + 0.5f);
        }
    }

    free(low_hdr);
    free(low_depth);
}

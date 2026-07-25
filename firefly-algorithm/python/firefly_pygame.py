"""
================================================================================
 FIREFLY ALGORITHM - Interactive Pygame Visualization
================================================================================
Companion to: "Bio-Inspired Artificial Intelligence (AI) Algorithms in K-12 and
Postgraduate STEM Education:The Firefly Algorithm Across Block-Based and
Textual Programming Environments"


 This module is the interactive, real-time counterpart to the analytical
 Python module (firefly_pedagogical.py) and to the MIT Scratch project. It
 gives upper-secondary / undergraduate / postgraduate learners the same kind
 of hands-on, parameter-manipulable experience that the Scratch project gives
 K-12 learners, but in a textual language, and it closes a gap the Scratch
 project deliberately leaves open: Scratch's LightSource is never driven by
 an objective function, so it demonstrates swarm *tracking*, not *convergence*
 (see Table 2 of the article). This program provides both.

 TWO MODES (press T to toggle):

   FREE-ROAM mode   - direct analogue of the Scratch project. A single
                       LightSource-equivalent target can be dragged with the
                       mouse, jumped to a random position (SPACE), or set
                       wandering continuously (M) - exactly the three
                       behaviours implemented in the Scratch LightSource
                       sprite. Fireflies chase it using the single-attractor
                       rule (FireflyAlgorithmSimplified). There is no fitness
                       function in this mode: it demonstrates tracking only.

   OPTIMIZE mode    - a real fitness landscape (Sphere, Rastrigin, or
                       Ackley, cycled with B) is drawn as a background heat
                       map, and the fireflies perform genuine Firefly
                       Algorithm search on it. Toggle the topology with C:
                         - Simplified: one shared light, updated automatically
                           to the best position found so far (mirrors
                           FireflyAlgorithmSimplified).
                         - Canonical:  full pairwise comparison, O(n^2)
                           (mirrors FireflyAlgorithmCanonical); watch for the
                           sub-grouping behaviour described in Section 4.1 of
                           the article, which the single-attractor topology
                           cannot show.

 CONTROLS
   Mouse drag on sliders : adjust FA_Vision, FA_Alpha, FA_Attraction,
                            FA_Absorption, FA_Fireflies live
   Mouse drag on target   : (Free-Roam only) reposition the light by hand
   SPACE                  : (Free-Roam only) jump light to a random position
   M                       : (Free-Roam only) toggle autonomous wandering
   T                       : toggle Free-Roam / Optimize mode
   C                       : (Optimize only) toggle Simplified / Canonical
   B                       : (Optimize only) cycle Sphere / Rastrigin / Ackley
   R                       : reset/respawn the swarm
   ESC / window close      : quit

 Requires: pygame, numpy
================================================================================
"""

import sys
import math
import random
import numpy as np
import pygame

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 900, 640
PLOT_RECT = pygame.Rect(20, 90, 640, 480)      # main swarm/landscape viewport
HUD_X = 680                                     # slider panel x-origin
FPS = 60

DOMAIN_LO, DOMAIN_HI = -5.12, 5.12              # Optimize-mode search domain
FREE_BOUNDS = PLOT_RECT.inflate(-10, -10)       # Free-Roam pixel bounds

BG_COLOR = (12, 14, 24)
PANEL_COLOR = (24, 27, 40)
TEXT_COLOR = (225, 228, 235)
MUTED_COLOR = (140, 148, 165)
DIM_COLOR = (68, 68, 255)
BRIGHT_CORE = (255, 255, 255)
BRIGHT_GLOW = (255, 224, 0)
LIGHT_COLOR = (255, 210, 60)


# --------------------------------------------------------------------------
# Objective functions (identical to firefly_pedagogical.py)
# --------------------------------------------------------------------------
def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley(x):
    n = len(x)
    s1 = np.sum(x ** 2); s2 = np.sum(np.cos(2 * np.pi * x))
    return (-20 * np.exp(-0.2 * np.sqrt(s1 / n))
            - np.exp(s2 / n) + 20 + np.e)


BENCHMARKS = {"Sphere": sphere, "Rastrigin": rastrigin, "Ackley": ackley}


# --------------------------------------------------------------------------
# Coordinate mapping: domain space <-> screen pixels (Optimize mode only)
# --------------------------------------------------------------------------
def domain_to_screen(dx, dy):
    sx = PLOT_RECT.left + (dx - DOMAIN_LO) / (DOMAIN_HI - DOMAIN_LO) * PLOT_RECT.width
    sy = PLOT_RECT.top + (dy - DOMAIN_LO) / (DOMAIN_HI - DOMAIN_LO) * PLOT_RECT.height
    return sx, sy


def screen_to_domain(sx, sy):
    dx = DOMAIN_LO + (sx - PLOT_RECT.left) / PLOT_RECT.width * (DOMAIN_HI - DOMAIN_LO)
    dy = DOMAIN_LO + (sy - PLOT_RECT.top) / PLOT_RECT.height * (DOMAIN_HI - DOMAIN_LO)
    return dx, dy


def build_heatmap_surface(objective_func, resolution=140):
    """Pre-render the fitness landscape once per (function, size) as a
    pygame Surface using a hand-rolled viridis-like colormap, matching the
    contour plot style used in Figure 5 of the article."""
    g = np.linspace(DOMAIN_LO, DOMAIN_HI, resolution)
    X, Y = np.meshgrid(g, g)
    Z = np.array([[objective_func(np.array([x, y])) for x, y in zip(rx, ry)]
                  for rx, ry in zip(X, Y)])
    Z = np.sqrt(Z)  # compress dynamic range for a more informative gradient
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)

    # Simple 4-stop viridis-ish colormap (dark purple -> teal -> green -> yellow)
    stops = [(0.0, (68, 1, 84)), (0.35, (59, 82, 139)),
             (0.65, (33, 145, 140)), (0.85, (94, 201, 98)), (1.0, (253, 231, 37))]

    def colormap(v):
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= v <= t1:
                f = (v - t0) / (t1 - t0 + 1e-9)
                return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
        return stops[-1][1]

    small = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for i in range(resolution):
        for j in range(resolution):
            small[j, i] = colormap(Z[i, j])
    surf = pygame.surfarray.make_surface(small)
    surf = pygame.transform.smoothscale(surf, (PLOT_RECT.width, PLOT_RECT.height))
    return surf


# --------------------------------------------------------------------------
# Firefly agent
# --------------------------------------------------------------------------
class Firefly:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.brightness = 0.0   # 0 (dim) .. 1 (bright), for drawing only

    def draw(self, screen):
        glow_r = 4 + 10 * self.brightness
        if self.brightness > 0.55:
            glow = pygame.Surface((int(glow_r * 4), int(glow_r * 4)), pygame.SRCALPHA)
            cx = cy = glow.get_width() // 2
            for rr, a in [(glow_r * 2, 40), (glow_r * 1.3, 90), (glow_r * 0.6, 255)]:
                pygame.draw.circle(glow, (*BRIGHT_GLOW, a), (cx, cy), int(rr))
            pygame.draw.circle(glow, (*BRIGHT_CORE, 235), (cx, cy), int(glow_r * 0.3))
            screen.blit(glow, (self.x - cx, self.y - cy), special_flags=pygame.BLEND_PREMULTIPLIED)
        else:
            shade = int(60 + 120 * self.brightness)
            pygame.draw.circle(screen, (shade // 2, shade // 2, min(255, shade + 60)),
                                (int(self.x), int(self.y)), int(3 + 3 * self.brightness))


# --------------------------------------------------------------------------
# Slider widget (on-screen analogue of the Scratch variable monitors)
# --------------------------------------------------------------------------
class Slider:
    def __init__(self, x, y, w, lo, hi, value, label, integer=False):
        self.rect = pygame.Rect(x, y, w, 14)
        self.lo, self.hi = lo, hi
        self.value = value
        self.label = label
        self.integer = integer
        self.dragging = False

    def handle_pos_x(self):
        t = (self.value - self.lo) / (self.hi - self.lo)
        return self.rect.left + int(t * self.rect.width)

    def hit(self, pos):
        handle = pygame.Rect(self.handle_pos_x() - 8, self.rect.centery - 8, 16, 16)
        return handle.collidepoint(pos) or self.rect.collidepoint(pos)

    def set_from_mouse(self, mx):
        t = max(0.0, min(1.0, (mx - self.rect.left) / self.rect.width))
        v = self.lo + t * (self.hi - self.lo)
        self.value = round(v) if self.integer else round(v, 4)

    def draw(self, screen, font):
        pygame.draw.rect(screen, (60, 64, 80), self.rect, border_radius=6)
        hx = self.handle_pos_x()
        pygame.draw.circle(screen, LIGHT_COLOR, (hx, self.rect.centery), 8)
        val_str = f"{self.value:g}"
        txt = font.render(f"{self.label}: {val_str}", True, TEXT_COLOR)
        screen.blit(txt, (self.rect.left, self.rect.top - 18))


# --------------------------------------------------------------------------
# Main program
# --------------------------------------------------------------------------
def main(headless_frames=None, start_mode="FREE", start_topology="Simplified", start_bench="Rastrigin",
         screenshot_path=None, screenshot_at=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Firefly Algorithm - Interactive (Free-Roam / Optimize)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 15)
    big_font = pygame.font.SysFont("Arial", 20, bold=True)

    # ---- state mirroring the Scratch/Python variable names -------------
    sliders = {
        "FA_Vision":     Slider(HUD_X, 140, 190, 10, 200, 80, "FA_Vision"),
        "FA_Alpha":      Slider(HUD_X, 190, 190, 1, 20, 5, "FA_Alpha"),
        "FA_Attraction": Slider(HUD_X, 240, 190, 0.05, 0.9, 0.3, "FA_Attraction"),
        "FA_Absorption": Slider(HUD_X, 290, 190, 0.0, 0.02, 0.001, "FA_Absorption"),
        "FA_Fireflies":  Slider(HUD_X, 340, 190, 10, 150, 30, "FA_Fireflies", integer=True),
    }

    mode = start_mode             # "FREE" or "OPTIMIZE"
    topology = start_topology     # "Simplified" or "Canonical" (Optimize mode only)
    bench_name = start_bench
    heatmaps = {name: build_heatmap_surface(fn) for name, fn in BENCHMARKS.items()}

    def spawn_fireflies(n):
        flies = []
        for _ in range(int(n)):
            if mode == "FREE":
                x = random.uniform(FREE_BOUNDS.left, FREE_BOUNDS.right)
                y = random.uniform(FREE_BOUNDS.top, FREE_BOUNDS.bottom)
            else:
                x, y = domain_to_screen(random.uniform(DOMAIN_LO, DOMAIN_HI),
                                         random.uniform(DOMAIN_LO, DOMAIN_HI))
            flies.append(Firefly(x, y))
        return flies

    fireflies = spawn_fireflies(sliders["FA_Fireflies"].value)

    # Free-Roam light state (direct analogue of the Scratch LightSource sprite)
    light_pos = [PLOT_RECT.centerx, PLOT_RECT.centery]
    wandering = False
    wander_dir = random.uniform(0, 2 * math.pi)
    dragging_light = False

    # Optimize-mode bookkeeping
    best_value = math.inf
    best_pos_screen = list(light_pos)
    iteration = 0

    active_slider = None
    running = True
    frame_count = 0

    while running:
        # ---------------- events ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:
                    mode = "OPTIMIZE" if mode == "FREE" else "FREE"
                    fireflies = spawn_fireflies(sliders["FA_Fireflies"].value)
                    best_value = math.inf
                elif event.key == pygame.K_c and mode == "OPTIMIZE":
                    topology = "Canonical" if topology == "Simplified" else "Simplified"
                elif event.key == pygame.K_b and mode == "OPTIMIZE":
                    bench_names = list(BENCHMARKS.keys())
                    bench_name = bench_names[(bench_names.index(bench_name) + 1) % len(bench_names)]
                    best_value = math.inf
                elif event.key == pygame.K_r:
                    fireflies = spawn_fireflies(sliders["FA_Fireflies"].value)
                    best_value = math.inf
                    iteration = 0
                elif event.key == pygame.K_SPACE and mode == "FREE":
                    light_pos[0] = random.uniform(FREE_BOUNDS.left, FREE_BOUNDS.right)
                    light_pos[1] = random.uniform(FREE_BOUNDS.top, FREE_BOUNDS.bottom)
                elif event.key == pygame.K_m and mode == "FREE":
                    wandering = not wandering
                    wander_dir = random.uniform(0, 2 * math.pi)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                hit_slider = False
                for s in sliders.values():
                    if s.hit(event.pos):
                        active_slider = s
                        s.set_from_mouse(event.pos[0])
                        hit_slider = True
                        break
                if not hit_slider and mode == "FREE":
                    if math.hypot(event.pos[0] - light_pos[0], event.pos[1] - light_pos[1]) < 20:
                        dragging_light = True
            elif event.type == pygame.MOUSEBUTTONUP:
                active_slider = None
                dragging_light = False
            elif event.type == pygame.MOUSEMOTION:
                if active_slider is not None:
                    active_slider.set_from_mouse(event.pos[0])
                    if active_slider.label == "FA_Fireflies":
                        fireflies = spawn_fireflies(active_slider.value)
                elif dragging_light:
                    light_pos[0] = min(max(event.pos[0], FREE_BOUNDS.left), FREE_BOUNDS.right)
                    light_pos[1] = min(max(event.pos[1], FREE_BOUNDS.top), FREE_BOUNDS.bottom)

        FA_Vision = sliders["FA_Vision"].value
        FA_Alpha = sliders["FA_Alpha"].value
        FA_Attraction = sliders["FA_Attraction"].value
        FA_Absorption = sliders["FA_Absorption"].value

        # ---------------- update ----------------
        if mode == "FREE":
            if wandering:
                wander_dir += random.uniform(-0.15, 0.15)
                light_pos[0] += 2.0 * math.cos(wander_dir)
                light_pos[1] += 2.0 * math.sin(wander_dir)
                if not FREE_BOUNDS.collidepoint(light_pos):
                    wander_dir += math.pi
                light_pos[0] = min(max(light_pos[0], FREE_BOUNDS.left), FREE_BOUNDS.right)
                light_pos[1] = min(max(light_pos[1], FREE_BOUNDS.top), FREE_BOUNDS.bottom)

            for fly in fireflies:
                dx = light_pos[0] - fly.x
                dy = light_pos[1] - fly.y
                dist = math.hypot(dx, dy)
                brightness_raw = max(0.0, 200 - dist)
                fly.brightness = min(1.0, brightness_raw / 200)
                found_brighter = dist < FA_Vision and brightness_raw < 180
                if found_brighter:
                    beta = FA_Attraction * math.exp(-FA_Absorption * dist * dist)
                    fly.x += beta * dx + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                    fly.y += beta * dy + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                else:
                    fly.x += random.uniform(-FA_Alpha, FA_Alpha)
                    fly.y += random.uniform(-FA_Alpha, FA_Alpha)
                fly.x = min(max(fly.x, FREE_BOUNDS.left), FREE_BOUNDS.right)
                fly.y = min(max(fly.y, FREE_BOUNDS.top), FREE_BOUNDS.bottom)

        else:  # OPTIMIZE mode - real search on the fitness landscape
            objective = BENCHMARKS[bench_name]
            gamma = FA_Absorption * 40          # rescaled for domain-space distances
            domain_positions = [screen_to_domain(f.x, f.y) for f in fireflies]
            values = [objective(np.array(p)) for p in domain_positions]

            if topology == "Simplified":
                bi = int(np.argmin(values))
                if values[bi] < best_value:
                    best_value = values[bi]
                    best_pos_screen = [fireflies[bi].x, fireflies[bi].y]
                for k, fly in enumerate(fireflies):
                    dx = best_pos_screen[0] - fly.x
                    dy = best_pos_screen[1] - fly.y
                    dist_px = math.hypot(dx, dy)
                    dist_dom = dist_px / PLOT_RECT.width * (DOMAIN_HI - DOMAIN_LO)
                    beta = FA_Attraction * math.exp(-gamma * dist_dom * dist_dom)
                    fly.x += beta * dx + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                    fly.y += beta * dy + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                    fly.brightness = 1.0 / (1.0 + values[k])
            else:  # Canonical: full pairwise O(n^2), reveals sub-grouping
                n = len(fireflies)
                for i in range(n):
                    for j in range(n):
                        if values[j] < values[i]:
                            dx = fireflies[j].x - fireflies[i].x
                            dy = fireflies[j].y - fireflies[i].y
                            dist_px = math.hypot(dx, dy)
                            dist_dom = dist_px / PLOT_RECT.width * (DOMAIN_HI - DOMAIN_LO)
                            beta = FA_Attraction * math.exp(-gamma * dist_dom * dist_dom)
                            fireflies[i].x += beta * dx + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                            fireflies[i].y += beta * dy + random.uniform(-FA_Alpha / 2, FA_Alpha / 2)
                            values[i] = objective(np.array(screen_to_domain(fireflies[i].x, fireflies[i].y)))
                bi = int(np.argmin(values))
                if values[bi] < best_value:
                    best_value = values[bi]
                for k, fly in enumerate(fireflies):
                    fly.brightness = 1.0 / (1.0 + values[k])

            for fly in fireflies:
                fly.x = min(max(fly.x, PLOT_RECT.left), PLOT_RECT.right)
                fly.y = min(max(fly.y, PLOT_RECT.top), PLOT_RECT.bottom)
            iteration += 1

        # ---------------- draw ----------------
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WIDTH, 80))
        title = big_font.render("Firefly Algorithm - Interactive Visualization", True, TEXT_COLOR)
        screen.blit(title, (20, 15))
        mode_str = f"Mode: {mode}" + (f"  |  Topology: {topology}  |  Function: {bench_name}" if mode == "OPTIMIZE" else "  (direct analogue of the Scratch project)")
        screen.blit(font.render(mode_str, True, MUTED_COLOR), (20, 45))
        screen.blit(font.render("T: toggle mode   C: topology   B: function   SPACE: jump light   M: wander   R: reset",
                                 True, MUTED_COLOR), (20, 63))

        pygame.draw.rect(screen, (30, 33, 46), PLOT_RECT)
        if mode == "OPTIMIZE":
            screen.blit(heatmaps[bench_name], PLOT_RECT.topleft)
        pygame.draw.rect(screen, (70, 74, 90), PLOT_RECT, width=2)

        for fly in fireflies:
            fly.draw(screen)

        if mode == "FREE":
            pygame.draw.circle(screen, LIGHT_COLOR, (int(light_pos[0]), int(light_pos[1])), 9)
            pygame.draw.circle(screen, (255, 255, 255), (int(light_pos[0]), int(light_pos[1])), 4)
        else:
            info = font.render(f"iteration {iteration}   best f = {best_value:.4e}", True, TEXT_COLOR)
            screen.blit(info, (PLOT_RECT.left, PLOT_RECT.bottom + 8))

        pygame.draw.rect(screen, PANEL_COLOR, (HUD_X - 20, 100, WIDTH - HUD_X + 20, HEIGHT - 120), border_radius=8)
        for s in sliders.values():
            s.draw(screen, font)

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1
        if screenshot_at is not None and frame_count == screenshot_at and screenshot_path is not None:
            pygame.image.save(screen, screenshot_path)
        if headless_frames is not None and frame_count >= headless_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()

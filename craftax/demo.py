# %%
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv

env = CraftaxEnv(CraftaxEnv.default_static_params())
obs, state = env.reset(rng, env.default_params)


# %%
jitted_reset = jax.jit(env.reset)

# %%
obs, state = jitted_reset(rng, env.default_params)


# %%
from craftax_marl.game_logic import craftax_step

jitted_step = jax.jit(craftax_step)

# %%
import pygame
from craftax_marl.renderer import render_craftax_pixels
from craftax_marl.constants import BLOCK_PIXEL_SIZE_HUMAN, OBS_DIM, INVENTORY_OBS_HEIGHT
import numpy as np


class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)[0]
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))
        return pixels

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

# %%
pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
renderer = CraftaxRenderer(env, env.default_params, pixel_render_size=pixel_render_size)


# %%
def register_press():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key
                d = {
                    pygame.K_LEFT: 1,
                    pygame.K_RIGHT: 2,
                    pygame.K_DOWN: 4,
                    pygame.K_UP: 3,
                }
                if key in d:
                    return d[key]


# %%
while True:
    # Check for quit event
    if renderer.is_quit_requested():
        break

    # Check for key press events
    action1 = register_press()
    print("Registered action1", action1)
    action2 = register_press()
    print("Registered action2", action2)
    action3 = register_press()
    print("Registered action3", action3)
    action4 = register_press()
    print("Registered action4", action4)
    # actions = jnp.array([action1, action2])
    actions = jnp.array([action1, action2, action3, action4])

    state, _ = jitted_step(
        rng, state, actions, env.default_params, CraftaxEnv.default_static_params()
    )
    renderer.render(state)
    renderer.update()


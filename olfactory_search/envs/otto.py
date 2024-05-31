"""
    Reference: 
     - Paper: https://link.springer.com/article/10.1140/epje/s10189-023-00277-8
     - Code: https://github.com/auroreloisy/otto-benchmark
"""

from abc import abstractmethod
from collections import namedtuple

import numpy as np
import scipy

import pygame
import gymnasium as gym
from gymnasium import spaces

from .parameters import ParametersIsotropic, ParametersWindy
from .rendering import downsample, OdorCell

# from .rendering import RenderFrame

SingleState = namedtuple("SingleState", ["pos", "hit"])

ACTIONS_2D = [
    np.array([1, 0]),  # right
    np.array([0, 1]),  # up
    np.array([-1, 0]),  # left
    np.array([0, -1]),  # right
]

ACTIONS_3D = [
    np.array([1, 0, 0]),  # x plus 1
    np.array([-1, 0, 0]),  # x minus 1
    np.array([0, 1, 0]),  # y plus 1
    np.array([0, -1, 0]),  # y minus 1
    np.array([0, 0, 1]),  # z plus 1
    np.array([0, 0, -1]),  # z minus 1
]


class OlfactoryEnv(gym.Env):
    """ """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        num_dimensions: int,
        parameters,
        task: str | None = None,
        render_mode: str | None = None,
        screen_size: int | None = 640,
    ):
        assert task is None or task in ["guess", "reach"]
        
        self.num_dimensions = num_dimensions
        self.parameters = parameters
        self.task = task
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.tile_size = 32
        # self.rf = RenderFrame(self.parameters.grid_size)

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=0,
                    high=self.parameters.grid_size - 1,
                    shape=(num_dimensions,),
                    dtype=np.int64,
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )
        self._action_to_direction = ACTIONS_2D if num_dimensions == 2 else ACTIONS_3D
        self.action_space = spaces.Discrete(len(self._action_to_direction))

        self._state = None

        self.render_size = None
        self.window = None
        self.screen_size = 600
        self.clock = None

        self._tile_cache = {}
        self._render_cells = {}
        self.history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._render_cells = {(i, j): OdorCell(hit=-2) for i in range(self.parameters.grid_size) for j in range(self.parameters.grid_size) }
        self.history = []

        agent_location = self.np_random.integers(
            0, self.parameters.grid_size, size=2, dtype=np.int64
        )
        source_location = self.np_random.integers(
            0, self.parameters.grid_size, size=2, dtype=np.int64
        )

        if options is not None:
            if "agent_location" in options:
                assert self.num_dimensions == len(options["agent_location"])
                agent_location = np.array(options["agent_location"], dtype=np.int64)
            if "source_location" in options:
                assert self.num_dimensions == len(options["source_location"])
                source_location = np.array(options["source_location"], dtype=np.int64)

        self._state = {"agent": agent_location, "source": source_location}

        observation = self._observation(self._state["agent"])
        info = self._info()

        # record states for rendering
        self.history.append(SingleState(self._state["agent"], observation["hits"]))

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._state["agent"] = np.clip(
            self._state["agent"] + direction, 0, self.parameters.grid_size - 1
        )

        # An episode is done iff the agent has reached the source
        terminated = np.array_equal(self._state["agent"], self._state["source"])
        truncated = False  # use gymnasium.make([...[, max_episode_steps=Parameters.T_max) to handle episode truncation
        observation = self._observation(self._state["agent"])
        reward = 0 if observation["hits"] == -1 else -1  # Binary sparse rewards
        info = self._info()

        # record states for rendering
        self.history.append(SingleState(self._state["agent"], observation["hits"]))

        return observation, reward, terminated, truncated, info

    def render_tile(
        self,
        obj,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> np.ndarray:
        # Hash map lookup key for the cache
        key = tuple([tile_size])
        key = obj.encode() + key if obj else key

        if key in self._tile_cache:
            return self._tile_cache[key]

        if obj is not None:
            img = 255*np.ones(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
            obj.render(img)
            # Downsample the image to perform supersampling/anti-aliasing
            img = downsample(img, subdivs)
            # Cache the rendered tile
            self._tile_cache[key] = img
            return img

    def render_frame(
        self,
        tile_size: int,
    ) -> np.ndarray:
        # Compute the total grid size
        width_px = self.parameters.grid_size * tile_size
        height_px = self.parameters.grid_size * tile_size

        img = 255*np.ones(shape=(height_px, width_px, 3), dtype=np.uint8)

        history = [tuple(item[0]) for item in self.history]
        hits = [item[1] for item in self.history]

        self._render_cells[tuple(self._state["source"])] = OdorCell(0, "source")


        for idx, item in enumerate(history):
            i, j = item[0], item[1]
            cell = self._render_cells[(i, j)]
            last_state = None if idx == 0 else history[idx - 1]
            next_state = None if idx == len(history) - 1 else history[idx + 1]
            last_action = None if last_state is None else np.array(history[idx]) - np.array(last_state)
            next_action = None if next_state is None else -(np.array(next_state) - np.array(history[idx]))

            if idx == 0:
                cell.type = "start"
            elif idx == len(history) - 1:
                cell.type = "agent"
            else:
                cell.type = "odor" if cell.type != "start" else cell.type
            
            if last_action is not None:
                cell.add_action(last_action[0], last_action[1])

            if next_action is not None:
                cell.add_action(next_action[0], next_action[1])
            
            cell.hit = hits[idx]
            self._render_cells[(i, j)] = cell


        # Render the grid
        for j in range(0, self.parameters.grid_size):
            for i in range(0, self.parameters.grid_size):
                cell = self._render_cells[(i, j)]
                tile_img = self.render_tile(
                    cell,
                    tile_size=tile_size,
                )
                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def render(self):
        img = self.render_frame(self.tile_size)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            # text = self.mission
            text = "Here is the text"
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()

    @abstractmethod
    def _observation(self, loc=None):
        raise NotImplementedError

    @abstractmethod
    def _info(self):  # this should not be exposed to agent
        raise NotImplementedError


class Isotropic2D(OlfactoryEnv):
    def __init__(
        self, num_dimensions, parameters: ParametersIsotropic, render_mode=None
    ):
        super(Isotropic2D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self, loc):
        if np.array_equal(loc, self._state["source"]):
            hits = -1
        else:
            # Euclidean distance
            r = np.linalg.norm(self._state["source"] - loc)
            # eq: B5
            mu_r = (
                scipy.special.k0(r / self.parameters.lambda_over_delta_x)
                / scipy.special.k0(1)
            ) * self.parameters.mu0_Poisson
            # weights = ((mu_r ** self.parameters.h) * np.exp(-mu_r)) / scipy.special.factorial(self.parameters.h)  # should be same as scipy.stats.poisson.pmf
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):  # this should not be exposed to agent
        return {"source": self._state["source"]}


class Isotropic3D(OlfactoryEnv):
    def __init__(
        self, num_dimensions, parameters: ParametersIsotropic, render_mode=None
    ):
        super(Isotropic3D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self, loc):
        if np.array_equal(loc, self._state["source"]):
            hits = -1
        else:
            # Euclidean distance
            r = np.linalg.norm(self._state["source"] - loc)
            # eq: A1b with V=0
            mu_r = (
                (self.parameters.R_times_delta_t * self.parameters.delta_x_over_a)
                / (2 * r)
                * np.exp(-r / self.parameters.lambda_over_delta_x)
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}


class Windy2D(OlfactoryEnv):
    # TODO: hit recording abnormally high
    def __init__(self, num_dimensions, parameters: ParametersWindy, render_mode=None):
        super(Windy2D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self, loc):
        # The wind blows in the positive x-direction from Section B2
        if np.array_equal(loc, self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            # x_position is vector(r) * e_x
            x_position = loc[0] - self._state["source"][0]
            mu_r = (
                self.parameters.R_bar
                / r
                * np.exp(
                    self.parameters.V_bar * x_position / self.parameters.delta_x_over_a
                    - r / self.parameters.lambda_over_a
                )
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

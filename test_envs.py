import time

from olfactory_search.envs import (
    SMALLER_ISOTROPIC_DOMAIN,
    LARGER_ISOTROPIC_DOMAIN,
    SMALLER_WINDY_DOMAIN_WITH_DETECTION,
    SMALLER_WINDY_DOMAIN_WITHOUT_DETECTION,
)

import gymnasium as gym
import numpy as np

ENV_LIST = [
    "olfactory_search/Isotropic2D-v0",
    "olfactory_search/Isotropic3D-v0",
]
ENV_PARAMS = [
    SMALLER_ISOTROPIC_DOMAIN,
    LARGER_ISOTROPIC_DOMAIN,
]

"""
Test the environments with random actions for Isotropic 2D and 3D
"""
for env_name in ENV_LIST:
    for parameters in ENV_PARAMS:
        env = gym.make(
            env_name,
            num_dimensions=2,
            parameters=parameters,
            max_episode_steps=parameters.T_max,
            render_mode="human",
        )
        seed = None
        # Uncomment the following if you specify the original location
        options = None
        # options = {
        #     'agent_location': [2, 2],
        #     'source_location': [6, 6],
        # }

        start_time = time.time()
        observation, info = env.reset(seed=seed, options=options)
        # print(f"initial observation = {observation}, info = {info}")
        hit_recording = []
        for _ in range(10000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            # print(f"action = {action}, observation = {observation}, reward = {reward}, done = {terminated or truncated}, info = {info}")

            if terminated or truncated:
                # print(f"episode done")
                observation, info = env.reset(seed=seed, options=options)
                # print(f"\ninitial observation = {observation}, info = {info}")

            hit_recording.append(observation["hits"])

        env.close()

        print(
            "Time for simulation %s is %.2fs"
            % (env_name[-14:], time.time() - start_time)
        )
        print(
            "Hit recording 0: %.2f, 1: %.2f, 2: %.2f, 3: %.2f"
            % (
                np.sum(np.array(hit_recording) == 0),
                np.sum(np.array(hit_recording) == 1),
                np.sum(np.array(hit_recording) == 2),
                np.sum(np.array(hit_recording) == 3),
            )
        )


"""
Test the environments with random actions for Windy 2D
"""
for parameters in [SMALLER_WINDY_DOMAIN_WITH_DETECTION, SMALLER_WINDY_DOMAIN_WITHOUT_DETECTION,]:
    env = gym.make(
        "olfactory_search/Windy2D-v0",
        num_dimensions=2,
        parameters=parameters,
        max_episode_steps=parameters.T_max,
        render_mode="human",
    )
    seed = None
    # Uncomment the following if you specify the original location
    options = None
    options = {
        'agent_location': [10, 10],
        'source_location': [2, 2],
    }

    start_time = time.time()
    observation, info = env.reset(seed=seed, options=options)
    # print(f"initial observation = {observation}, info = {info}")
    hit_recording = []
    for _ in range(10000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # env.render()
        # print(f"action = {action}, observation = {observation}, reward = {reward}, done = {terminated or truncated}, info = {info}")

        if terminated or truncated:
            # print(f"episode done")
            observation, info = env.reset(seed=seed, options=options)
            # print(f"\ninitial observation = {observation}, info = {info}")

        hit_recording.append(observation["hits"])

    env.close()

    print(
        "Time for simulation %s is %.2fs"
        % ("Windy2D-v0", time.time() - start_time)
    )
    print(
        "Hit recording 0: %.2f, 1: %.2f, 2: %.2f, 3: %.2f"
        % (
            np.sum(np.array(hit_recording) == 0),
            np.sum(np.array(hit_recording) == 1),
            np.sum(np.array(hit_recording) == 2),
            np.sum(np.array(hit_recording) == 3),
        )
    )

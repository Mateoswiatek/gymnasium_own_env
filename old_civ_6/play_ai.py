# play_ai.py - AI agent playing the game
from MicroCivEnv import MicroCivEnv
import numpy as np
import random

from enums import ActionType


class SimpleAI:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        valid_actions = observation['valid_actions']
        valid_indices = np.where(valid_actions == 1)[0]

        # Simple strategy: prioritize building cities and creating settlers
        priority_actions = [
            ActionType.BUILD_CITY.value,
            ActionType.CREATE_SETTLER.value,
            ActionType.CREATE_WORKER.value,
            ActionType.BUILD_FARM.value,
            ActionType.BUILD_MINE.value,
            ActionType.BUILD_LUMBERMILL.value,
            ActionType.CREATE_WARRIOR.value,
            ActionType.MOVE_UP.value,
            ActionType.MOVE_DOWN.value,
            ActionType.MOVE_LEFT.value,
            ActionType.MOVE_RIGHT.value,
            ActionType.END_TURN.value,
            ActionType.DO_NOTHING.value
        ]

        for action in priority_actions:
            if action in valid_indices:
                return action

        # If no priority action is available, choose randomly
        return random.choice(valid_indices)

def main():
    env = MicroCivEnv(render_mode="human")
    ai = SimpleAI(env)
    observation, _ = env.reset()

    for _ in range(1000):  # Max episodes
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = ai.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated:
                print(f"Game over! Reason: {info.get('reason', 'Unknown')}")
                break

        observation, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()
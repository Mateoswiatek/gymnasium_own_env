import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union, Any
import pygame
import time
import math

class GraphGame:
    """
    A graph-based strategy game where players capture bases and send units between them.

    Game Rules:
    - Each vertex (base) belongs to a player or is neutral
    - Bases generate units each step up to a limit (50 for players, 10 for neutral)
    - Players can send units between bases
    - Units take time to travel between bases based on the edge weight
    - When enemy units meet on a path, they eliminate each other 1:1
    - When units reach a base, they are added to friendly bases or deducted from enemy bases
    - If units reduce an enemy base to negative units, the base is captured
    - The player who eliminates all others wins
    """

    def __init__(self, num_nodes: int, max_distance: float, num_players: int = 2,
                 neutral_bots: int = 2, seed: Optional[int] = None):
        """
        Initialize the game with the given parameters.

        Args:
            num_nodes: Total number of nodes/bases on the board
            max_distance: Maximum Euclidean distance between the furthest nodes
            num_players: Number of players in the game
            neutral_bots: Number of neutral bases
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.num_nodes = num_nodes
        self.max_distance = max_distance
        self.num_players = num_players
        self.neutral_bots = neutral_bots

        # Initialize graph
        self.graph = nx.Graph()

        # Node properties
        self.node_positions = {}  # (x, y) coordinates
        self.node_owners = {}     # Player ID who owns the node (0 for neutral)
        self.node_units = {}      # Number of units at each node

        # Units in transit
        self.moving_units = []  # List of dicts tracking units moving between nodes

        # Game state
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.step_count = 0

        # Initialize the board
        self._initialize_board()

    def _initialize_board(self):
        """Set up the initial board state with nodes, edges, and player distributions."""
        # Create nodes with random positions
        for i in range(self.num_nodes):
            # Random position within a square grid
            x = np.random.uniform(0, self.max_distance)
            y = np.random.uniform(0, self.max_distance)
            self.node_positions[i] = (x, y)
            self.graph.add_node(i)

        # Create edges between all nodes with weights based on Euclidean distance
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos_i = self.node_positions[i]
                pos_j = self.node_positions[j]
                # Calculate Euclidean distance and round to integer steps
                distance = int(np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2))
                # Ensure minimum distance of 1
                distance = max(1, distance)
                self.graph.add_edge(i, j, weight=distance)

        # Assign node ownership
        node_indices = list(range(self.num_nodes))
        np.random.shuffle(node_indices)

        # Assign neutral nodes
        for i in range(self.neutral_bots):
            if i < len(node_indices):
                self.node_owners[node_indices[i]] = 0  # 0 represents neutral
                self.node_units[node_indices[i]] = np.random.randint(5, 11)  # Start with 5-10 units

        # Assign player nodes
        player_nodes_per_player = (self.num_nodes - self.neutral_bots) // self.num_players
        for player_id in range(1, self.num_players + 1):
            start_idx = self.neutral_bots + (player_id - 1) * player_nodes_per_player
            end_idx = start_idx + player_nodes_per_player

            for i in range(start_idx, min(end_idx, len(node_indices))):
                self.node_owners[node_indices[i]] = player_id
                self.node_units[node_indices[i]] = np.random.randint(10, 21)  # Start with 10-20 units

        # Assign any remaining nodes
        for i in range(self.neutral_bots + player_nodes_per_player * self.num_players, len(node_indices)):
            player_id = np.random.randint(1, self.num_players + 1)
            self.node_owners[node_indices[i]] = player_id
            self.node_units[node_indices[i]] = np.random.randint(10, 21)

    def get_valid_actions(self, player_id: int) -> List[Tuple[int, int, int]]:
        """
        Get all valid actions for the given player.

        Returns:
            List of tuples (source_node, target_node, max_units) representing valid moves
        """
        valid_actions = []

        # Find nodes owned by the player
        player_nodes = [node for node, owner in self.node_owners.items() if owner == player_id]

        # For each node owned by the player, they can send units to any connected node
        for source in player_nodes:
            units_available = self.node_units[source]
            if units_available <= 0:
                continue

            for target in self.graph.neighbors(source):
                # Can send any number of units from 1 to all available
                valid_actions.append((source, target, units_available))

        return valid_actions

    def send_units(self, source: int, target: int, units: int, player_id: int) -> bool:
        """
        Send units from one node to another.

        Args:
            source: Source node ID
            target: Target node ID
            units: Number of units to send
            player_id: ID of the player making the move

        Returns:
            bool: Whether the move was valid and executed
        """
        # Check if move is valid
        if (self.node_owners[source] != player_id or
                not self.graph.has_edge(source, target) or
                units > self.node_units[source] or
                units <= 0):
            return False

        # Deduct units from source node
        self.node_units[source] -= units

        # Calculate arrival time based on edge weight
        travel_time = self.graph[source][target]['weight']

        # Add units to moving_units list
        self.moving_units.append({
            'source': source,
            'target': target,
            'units': units,
            'player_id': player_id,
            'steps_remaining': travel_time,
            'path_progress': 0.0  # For visualization
        })

        return True

    def _process_unit_movements(self):
        """Process all units currently in transit."""
        # Decrease remaining steps for all moving units
        for unit in self.moving_units:
            unit['steps_remaining'] -= 1
            # Update progress for visualization
            total_steps = self.graph[unit['source']][unit['target']]['weight']
            unit['path_progress'] = (total_steps - unit['steps_remaining']) / total_steps

        # Handle unit collisions on paths
        self._handle_unit_collisions()

        # Process units that have reached their destination
        arrived_units = [unit for unit in self.moving_units if unit['steps_remaining'] <= 0]
        for unit in arrived_units:
            self._process_unit_arrival(unit)
            self.moving_units.remove(unit)

    def _handle_unit_collisions(self):
        """Handle collisions between opposing units on the same path."""
        # Group units by path (source-target pair)
        paths = {}
        for unit in self.moving_units:
            path_key = tuple(sorted([unit['source'], unit['target']]))
            if path_key not in paths:
                paths[path_key] = {}

            player = unit['player_id']
            if player not in paths[path_key]:
                paths[path_key][player] = []

            paths[path_key][player].append(unit)

        # Check for collisions on each path
        for path, players_units in paths.items():
            if len(players_units) <= 1:  # Only one player's units on this path
                continue

            # Calculate progress along the path for each unit
            for player, units in players_units.items():
                for unit in units:
                    source, target = unit['source'], unit['target']
                    total_distance = self.graph[source][target]['weight']
                    # Normalize direction so we can compare positions
                    if path[0] == source:  # Moving from path[0] to path[1]
                        unit['normalized_pos'] = (total_distance - unit['steps_remaining']) / total_distance
                    else:  # Moving from path[1] to path[0]
                        unit['normalized_pos'] = unit['steps_remaining'] / total_distance

            # Sort all units by position on path
            all_units = []
            for player, units in players_units.items():
                all_units.extend(units)

            all_units.sort(key=lambda u: u['normalized_pos'])

            # Check for collisions between adjacent units of different players
            i = 0
            while i < len(all_units) - 1:
                unit1 = all_units[i]
                unit2 = all_units[i + 1]

                if (unit1['player_id'] != unit2['player_id'] and
                        abs(unit1['normalized_pos'] - unit2['normalized_pos']) < 0.1):  # Close enough to collide

                    # Units destroy each other 1:1
                    min_units = min(unit1['units'], unit2['units'])
                    unit1['units'] -= min_units
                    unit2['units'] -= min_units

                    # Remove units that have been completely destroyed
                    if unit1['units'] <= 0:
                        self.moving_units.remove(unit1)
                        all_units.pop(i)
                        # Don't increment i as we've removed an element
                    else:
                        i += 1

                    if unit2['units'] <= 0:
                        self.moving_units.remove(unit2)
                        all_units.pop(i)  # i is now pointing at the next element
                else:
                    i += 1

    def _process_unit_arrival(self, unit: Dict):
        """Process a unit that has reached its destination."""
        target = unit['target']
        player_id = unit['player_id']
        arriving_units = unit['units']

        if self.node_owners[target] == player_id:
            # Friendly node: add units
            self.node_units[target] += arriving_units
        else:
            # Enemy node: subtract units
            self.node_units[target] -= arriving_units

            # If units reduced to zero or below, capture the node
            if self.node_units[target] <= 0:
                old_owner = self.node_owners[target]
                self.node_owners[target] = player_id
                self.node_units[target] = abs(self.node_units[target])  # Remaining units stay

    def _generate_units(self):
        """Generate new units at each node based on ownership."""
        for node, owner in self.node_owners.items():
            # Skip nodes with no owner
            if owner == 0:  # neutral
                max_units = 10
            else:  # player owned
                max_units = 50

            # Generate one unit per step if below limit
            if self.node_units[node] < max_units:
                self.node_units[node] += 1

    def _check_game_over(self):
        """Check if the game is over (only one player remains)."""
        active_players = set(owner for owner in self.node_owners.values() if owner > 0)

        # Also check for players with units in transit
        for unit in self.moving_units:
            if unit['player_id'] > 0:  # Not neutral
                active_players.add(unit['player_id'])

        if len(active_players) <= 1:
            self.game_over = True
            if len(active_players) == 1:
                self.winner = next(iter(active_players))
            else:
                self.winner = 0  # Draw or all players eliminated

    def step(self):
        """Advance the game state by one step."""
        if self.game_over:
            return

        # Process unit movements
        self._process_unit_movements()

        # Generate new units
        self._generate_units()

        # Check for game over
        self._check_game_over()

        # Switch to next player
        self.current_player = (self.current_player % self.num_players) + 1

        # Increment step counter
        self.step_count += 1

    def render(self, mode='human'):
        """Render the current game state."""
        plt.figure(figsize=(10, 8))

        # Create a copy of the graph for visualization
        G = self.graph.copy()

        # Set node colors based on ownership
        node_colors = []
        for node in G.nodes():
            owner = self.node_owners[node]
            if owner == 0:  # Neutral
                node_colors.append('gray')
            else:
                # Use a different color for each player
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
                node_colors.append(colors[(owner - 1) % len(colors)])

        # Set node sizes based on unit count
        node_sizes = [100 + self.node_units[node] * 5 for node in G.nodes()]

        # Draw the graph
        pos = self.node_positions
        nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True,
                font_weight='bold', font_color='white')

        # Draw edge weights
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Draw units in transit
        for unit in self.moving_units:
            source_pos = pos[unit['source']]
            target_pos = pos[unit['target']]
            progress = unit['path_progress']

            # Calculate current position
            current_x = source_pos[0] + (target_pos[0] - source_pos[0]) * progress
            current_y = source_pos[1] + (target_pos[1] - source_pos[1]) * progress

            # Draw unit with color based on owner
            colors = ['gray', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
            color = colors[unit['player_id'] % len(colors)]

            plt.scatter(current_x, current_y, color=color, s=50 + unit['units'] * 2,
                        edgecolor='black', linewidth=1)
            plt.text(current_x, current_y, str(unit['units']),
                     horizontalalignment='center', verticalalignment='center')

        # Add game information
        plt.title(f"Step: {self.step_count}, Current Player: {self.current_player}")
        if self.game_over:
            if self.winner > 0:
                plt.suptitle(f"Game Over! Player {self.winner} wins!", fontsize=16)
            else:
                plt.suptitle("Game Over! It's a draw!", fontsize=16)

        plt.tight_layout()
        plt.show()

    def get_state(self):
        """Get the current state of the game as a dictionary."""
        return {
            'node_owners': self.node_owners.copy(),
            'node_units': self.node_units.copy(),
            'moving_units': self.moving_units.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'step_count': self.step_count
        }


class GraphGameEnv(gym.Env):
    """
    Gymnasium environment wrapper for the graph game.

    This environment follows the OpenAI Gym/Gymnasium interface for
    reinforcement learning.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, num_nodes=10, max_distance=100, num_players=2,
                 neutral_bots=2, render_mode='human', seed=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.max_distance = max_distance
        self.num_players = num_players
        self.neutral_bots = neutral_bots
        self.render_mode = render_mode
        self.seed_value = seed

        # Create the game
        self.game = GraphGame(num_nodes, max_distance, num_players, neutral_bots, seed)

        # Define action and observation spaces
        # Action: (source_node, target_node, units)
        self.action_space = spaces.Dict({
            'source': spaces.Discrete(num_nodes),
            'target': spaces.Discrete(num_nodes),
            'units': spaces.Discrete(51),  # 0-50 units
        })

        # Observation: node owners, node units, and units in transit
        # This is a simplified version - a more sophisticated representation would be better for learning
        self.observation_space = spaces.Dict({
            'node_owners': spaces.Box(low=0, high=num_players, shape=(num_nodes,), dtype=np.int32),
            'node_units': spaces.Box(low=0, high=100, shape=(num_nodes,), dtype=np.int32),
            'current_player': spaces.Discrete(num_players + 1),  # 1 to num_players
        })

        # For visualization
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment to a new initial state."""
        if seed is not None:
            self.seed_value = seed

        # Create a new game instance
        self.game = GraphGame(self.num_nodes, self.max_distance,
                              self.num_players, self.neutral_bots, self.seed_value)

        # Get initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """
        Take a step in the environment with the given action.

        Args:
            action: Dict with keys 'source', 'target', and 'units'

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extract action components
        source = action['source']
        target = action['target']
        units = action['units']

        # Get current player
        player = self.game.current_player

        # Execute action
        valid_move = self.game.send_units(source, target, units, player)

        # If the move was invalid, penalize the agent
        if not valid_move:
            reward = -10  # Penalty for invalid move
            observation = self._get_observation()
            return observation, reward, False, False, {'valid_move': False}

        # Advance the game state
        self.game.step()

        # Check if game is over
        terminated = self.game.game_over

        # Calculate reward
        reward = self._calculate_reward(player)

        # Get new observation
        observation = self._get_observation()

        # Additional info
        info = {
            'valid_move': True,
            'player': player,
            'winner': self.game.winner if terminated else None
        }

        return observation, reward, terminated, False, info

    def _calculate_reward(self, player):
        """Calculate reward for the current player based on game state."""
        # Count nodes owned by player
        player_nodes = sum(1 for owner in self.game.node_owners.values() if owner == player)

        # Count total units owned by player
        player_units = sum(units for node, units in self.game.node_units.items()
                           if self.game.node_owners[node] == player)

        # Units in transit
        player_moving_units = sum(unit['units'] for unit in self.game.moving_units
                                  if unit['player_id'] == player)

        # Basic reward based on control and units
        reward = player_nodes * 2 + (player_units + player_moving_units) * 0.1

        # Bonus for capturing nodes (detect changes from previous state)
        # This would require tracking previous state

        # Bonus/penalty for winning/losing
        if self.game.game_over:
            if self.game.winner == player:
                reward += 100  # Big bonus for winning
            else:
                reward -= 50   # Penalty for losing

        return reward

    def _get_observation(self):
        """Convert the game state to the observation format."""
        # Convert dictionaries to arrays for the observation space
        node_owners = np.array([self.game.node_owners.get(i, 0) for i in range(self.num_nodes)])
        node_units = np.array([self.game.node_units.get(i, 0) for i in range(self.num_nodes)])

        return {
            'node_owners': node_owners,
            'node_units': node_units,
            'current_player': self.game.current_player
        }

    def render(self):
        """Render the game state."""
        if self.render_mode == 'human':
            self.game.render()
            time.sleep(0.2)  # Small delay to see the rendering

        # Could implement 'rgb_array' mode for returning pixel representation

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Example usage:
if __name__ == "__main__":
    # Create a game with 7 nodes, max distance 100, 2 players and 1 neutral bot
    game = GraphGame(num_nodes=7, max_distance=100, num_players=2, neutral_bots=1, seed=42)

    # Display initial state
    game.render()

    # Example gameplay loop
    while not game.game_over:
        # Get valid moves for current player
        valid_moves = game.get_valid_actions(game.current_player)

        if valid_moves:
            # Select a random valid move
            move = valid_moves[np.random.choice(len(valid_moves))]
            source, target, max_units = move
            # Randomly choose how many units to send (between 1 and max)
            units_to_send = np.random.randint(1, max_units + 1)

            print(f"Player {game.current_player} sends {units_to_send} units from node {source} to node {target}")
            game.send_units(source, target, units_to_send, game.current_player)

        # Advance game state
        game.step()

        # Render updated state
        game.render()

    print(f"Game over! Winner: Player {game.winner}")

    # Example of using as a Gymnasium environment
    print("\nTesting as Gymnasium environment:")
    env = GraphGameEnv(num_nodes=7, max_distance=100, num_players=2, neutral_bots=1)
    obs, info = env.reset(seed=42)

    terminated = False
    while not terminated:
        # Random action
        action = {
            'source': np.random.randint(0, env.num_nodes),
            'target': np.random.randint(0, env.num_nodes),
            'units': np.random.randint(1, 11)  # 1-10 units
        }

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Reward: {reward}, Game Over: {terminated}")

        if not info['valid_move']:
            print("Invalid move!")

    if info['winner']:
        print(f"Winner: Player {info['winner']}")
    else:
        print("No winner (draw)")

    env.close()
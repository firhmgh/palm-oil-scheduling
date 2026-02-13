import numpy as np
import copy

class PalmEnv:
    """
    Lingkungan simulasi panen dan logistik TBS
    Mendukung reward shaping bertahap untuk MAPPO
    """

    def __init__(self, grid_size=10, max_teams=5, trucks=5, slots=6, domain_random=True):
        self.grid_size = grid_size
        self.max_teams = max_teams
        self.trucks = trucks
        self.slots = slots
        self.domain_random = domain_random
        self.max_time = 12 * 60  # 12 jam (menit)
        self.reset()

    def reset(self):
        self.time = 0

        self.blocks = [{
            'ready_fruit': np.random.randint(5, 20),
            'age': np.random.randint(1, 10),
            'distance': np.random.randint(1, 20)
        } for _ in range(self.grid_size ** 2)]

        self.truck_pos = [0] * self.trucks
        self.plant_slots = [0] * self.slots

        # Domain randomization
        if self.domain_random:
            self.weather = np.random.choice(['sunny', 'rain', 'heavy_rain'])
            self.road_condition = np.random.choice(['good', 'moderate', 'bad'])
        else:
            self.weather = 'sunny'
            self.road_condition = 'good'

        return self._get_state()

    def _get_state(self):
        return {
            'blocks': copy.deepcopy(self.blocks),
            'trucks': self.truck_pos.copy(),
            'slots': self.plant_slots.copy(),
            'time': self.time,
            'weather': self.weather,
            'road': self.road_condition
        }

    def compute_reward(self, prev_state, curr_state):
        """
        Reward shaping bertahap:
        - Produktivitas
        - Penalti antrian
        - Penalti idle
        - Penalti waktu
        """

        # 1️⃣ Reward panen
        prev_fruit = sum(b['ready_fruit'] for b in prev_state['blocks'])
        curr_fruit = sum(b['ready_fruit'] for b in curr_state['blocks'])
        harvest_reward = prev_fruit - curr_fruit

        # 2️⃣ Penalti antrian pabrik
        queue_penalty = -0.1 * sum(curr_state['slots'])

        # 3️⃣ Penalti idle block
        idle_blocks = sum(1 for b in curr_state['blocks'] if b['ready_fruit'] == 0)
        idle_penalty = -0.05 * idle_blocks

        # 4️⃣ Penalti waktu (efisiensi)
        time_penalty = -0.01

        total_reward = (
            harvest_reward +
            queue_penalty +
            idle_penalty +
            time_penalty
        )

        return total_reward

    def step(self, action_dict):
        prev_state = self._get_state()

        # === Scheduler ===
        if 'scheduler' in action_dict:
            block_idx = action_dict['scheduler']
            block_idx = int(np.clip(block_idx, 0, len(self.blocks) - 1))
            harvested = min(self.max_teams, self.blocks[block_idx]['ready_fruit'])
            self.blocks[block_idx]['ready_fruit'] -= harvested

        # === Simulasi antrian pabrik ===
        for i in range(self.slots):
            self.plant_slots[i] = np.random.randint(0, 3)

        self.time += 1
        done = self.time >= self.max_time

        curr_state = self._get_state()
        reward = self.compute_reward(prev_state, curr_state)

        return curr_state, reward, done, {}
    
    def get_observations(self):
        """
        Mengembalikan observasi TERPISAH untuk tiap agen (MAPPO style)
        """
        # === Scheduler melihat kondisi blok ===
        scheduler_obs = np.array([
            np.mean([b["ready_fruit"] for b in self.blocks]),
            np.mean([b["age"] for b in self.blocks]),
            np.mean([b["distance"] for b in self.blocks])
        ], dtype=np.float32)

        # === Dispatcher melihat posisi truk & jarak rata-rata ===
        dispatcher_obs = np.array([
            np.mean(self.truck_pos),
            np.mean([b["distance"] for b in self.blocks]),
            len(self.blocks)
        ], dtype=np.float32)

        # === Plant controller melihat antrian pabrik ===
        plant_obs = np.array([
            np.mean(self.plant_slots),
            np.max(self.plant_slots),
            np.sum(self.plant_slots)
        ], dtype=np.float32)

        return {
            "scheduler": scheduler_obs,
            "dispatcher": dispatcher_obs,
            "plant": plant_obs
        }

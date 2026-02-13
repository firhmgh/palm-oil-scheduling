import numpy as np
import copy

class PalmEnv:
    """
    Lingkungan simulasi panen dan logistik TBS Berbasis Multi-Agen.
    Memenuhi standar: 10x10 grid, 12 jam horizon, 3 level agen.
    """

    def __init__(self, grid_size=10, max_teams=10, trucks=10, slots=6, domain_random=True):
        self.grid_size = grid_size
        self.max_teams = max_teams
        self.trucks = trucks
        self.slots = slots
        self.domain_random = domain_random
        self.max_time = 12 * 60  # 720 menit (06.00 - 18.00)
        
        # Konstanta Kapasitas 
        self.plant_capacity_per_min = 2.0  # Ton per menit (Misal: 120 ton/jam)
        self.truck_capacity = 8.0          # Ton per truk
        self.optimal_age_window = 24       # Jam window panen optimal
        
        self.reset()

    def reset(self):
        self.time = 0
        
        # Inisialisasi Metrik (Untuk Evaluasi Target Kinerja)
        self.metrics = {
            'total_delay_minutes': 0,
            'total_throughput_tons': 0,
            'total_operation_cost': 0,
            'avg_queue_time': 0,
            'harvested_tons': 0,
            'queue_count': []
        }

        # Inisialisasi Blok (Grid 10x10)
        # Beda umur panen, jarak, dan produktivitas
        self.blocks = []
        for i in range(self.grid_size ** 2):
            self.blocks.append({
                'id': i,
                'ready_fruit': np.random.uniform(5, 50), # Ton
                'age': np.random.randint(12, 48),       # Jam sejak matang
                'distance': np.random.randint(5, 30),    # KM ke pabrik
                'is_closed': False                       # Jika jalan rusak/hujan deras
            })

        # Inisialisasi Status Truk
        self.truck_states = []
        for _ in range(self.trucks):
            self.truck_states.append({
                'pos': 0,           # 0 = Pabrik, 1+ = Block ID
                'load': 0,          # Isi ton saat ini
                'status': 'idle',   # idle, moving_to_block, loading, moving_to_plant, queueing
                'remaining_time': 0 # Waktu sisa perjalanan/proses
            })

        self.plant_queue = [] # List berisi berat (ton) dalam antrian
        
        # Inisialisasi Cuaca (Stochasticity & Domain Randomization)
        self._update_weather()
        
        return self.get_observations()

    def _update_weather(self):
        """Update kondisi cuaca yang mempengaruhi kecepatan (Stochasticity)"""
        if self.domain_random:
            # Peluang berubah setiap saat atau interval tertentu
            self.weather = np.random.choice(['sunny', 'rain', 'heavy_rain'], p=[0.7, 0.2, 0.1])
        else:
            self.weather = 'sunny'
            
        # Faktor kecepatan (Multiplier)
        if self.weather == 'sunny':
            self.speed_multiplier = 1.0
            self.road_closure_prob = 0.0
        elif self.weather == 'rain':
            self.speed_multiplier = 0.6  # Kecepatan truk turun 40%
            self.road_closure_prob = 0.05
        else: # heavy_rain
            self.speed_multiplier = 0.3  # Kecepatan truk turun 70%
            self.road_closure_prob = 0.2 # Potensi jalan tertutup 

    def step(self, action_dict):
        """
        Transisi dinamis per 1 menit
        """
        prev_state = self._get_state_for_reward()
        
        # 1. Update Cuaca setiap 3 jam (180 menit)
        if self.time % 180 == 0:
            self._update_weather()

        # 2. Proses Action Agen 1: Scheduler (Pilih Blok & Tim)
        if 'scheduler' in action_dict:
            action = action_dict['scheduler']
            block_idx = int(action % (self.grid_size**2))
            num_teams = int(action // (self.grid_size**2)) + 1
            
            # Operasi Panen
            harvest_rate = 0.5 * num_teams # Misal: 0.5 ton/menit per tim
            tonnage = min(self.blocks[block_idx]['ready_fruit'], harvest_rate)
            self.blocks[block_idx]['ready_fruit'] -= tonnage
            self.metrics['harvest_reward_temp'] = tonnage
            self.metrics['total_operation_cost'] += (num_teams * 10) # Biaya tim

        # 3. Proses Action Agen 2: Dispatcher (Alokasi Truk)
        if 'dispatcher' in action_dict:
            truck_idx = action_dict['dispatcher'] % self.trucks
            if self.truck_states[truck_idx]['status'] == 'idle':
                # Alokasikan truk ke blok yang butuh angkut (logika disederhanakan)
                target_block = np.argmax([b['ready_fruit'] for b in self.blocks])
                dist = self.blocks[target_block]['distance']
                
                self.truck_states[truck_idx]['status'] = 'moving_to_block'
                # Waktu tempuh dipengaruhi cuaca (speed_multiplier)
                self.truck_states[truck_idx]['remaining_time'] = int(dist / (0.8 * self.speed_multiplier))
                self.metrics['total_operation_cost'] += 5 # Biaya bahan bakar

        # 4. Update Simulasi Fisik (Truk & Antrean)
        for truck in self.truck_states:
            if truck['remaining_time'] > 0:
                truck['remaining_time'] -= 1
            else:
                if truck['status'] == 'moving_to_block':
                    truck['status'] = 'loading'
                    truck['remaining_time'] = 15 # Waktu muat 15 menit
                elif truck['status'] == 'loading':
                    truck['status'] = 'moving_to_plant'
                    truck['load'] = self.truck_capacity
                    truck['remaining_time'] = 20 # Waktu balik ke pabrik
                elif truck['status'] == 'moving_to_plant':
                    truck['status'] = 'queueing'
                    self.plant_queue.append(truck['load'])
                    truck['load'] = 0
                    truck['status'] = 'idle'

        # 5. Proses Action Agen 3: Plant Controller (Proses Antrean)
        # Kapasitas olah pabrik
        processed = 0
        if self.plant_queue:
            processed = min(sum(self.plant_queue), self.plant_capacity_per_min)
            # Pengurangan antrean (sederhana)
            if len(self.plant_queue) > 0:
                self.plant_queue[0] -= processed
                if self.plant_queue[0] <= 0:
                    self.plant_queue.pop(0)
        
        self.metrics['total_throughput_tons'] += processed
        self.metrics['queue_count'].append(len(self.plant_queue))

        # 6. Update Waktu & Penalti Keterlambatan
        for b in self.blocks:
            b['age'] += (1/60) # Bertambah umur tiap menit
            if b['age'] > self.optimal_age_window:
                # Penalti per menit per ton 
                delay_penalty = (b['ready_fruit'] * 0.8)
                self.metrics['total_delay_minutes'] += delay_penalty

        self.time += 1
        done = self.time >= self.max_time
        
        # Kirim info metrik untuk logging
        info = copy.deepcopy(self.metrics)
        info['avg_queue_time'] = np.mean(self.metrics['queue_count']) if self.metrics['queue_count'] else 0
        
        curr_state = self._get_state_for_reward()
        reward = self.compute_reward(prev_state, curr_state)

        return self.get_observations(), reward, done, info

    def _get_state_for_reward(self):
        """Internal state untuk hitung reward"""
        return {
            'ready_fruit': sum(b['ready_fruit'] for b in self.blocks),
            'queue_len': len(self.plant_queue),
            'weather': self.weather
        }

    def compute_reward(self, prev_state, curr_state):
        """Reward Shaping Bertahap sesuai Arsitektur MARL"""
        # Reward 1: Throughput (Semakin banyak ton diproses, semakin baik)
        throughput_reward = (curr_state['ready_fruit'] < prev_state['ready_fruit']) * 1.0
        
        # Penalti 2: Antrean Berlebih (Target: < 45 menit)
        queue_penalty = -0.5 if curr_state['queue_len'] > 5 else 0
        
        # Penalti 3: Operasional (Cost)
        cost_penalty = -0.1
        
        # Bonus 4: Ketepatan waktu (Jika panen di jendela optimal)
        # (Dihitung dari selisih keterlambatan di metrics)
        
        return throughput_reward + queue_penalty + cost_penalty

    def get_observations(self):
        """Observasi terpisah (Decentralized Execution)"""
        # Scheduler: Fokus pada buah dan umur
        obs_scheduler = np.array([
            sum(b['ready_fruit'] for b in self.blocks),
            np.mean([b['age'] for b in self.blocks]),
            1 if self.weather == 'heavy_rain' else 0
        ], dtype=np.float32)

        # Dispatcher: Fokus pada posisi truk dan kemacetan
        obs_dispatcher = np.array([
            sum(1 for t in self.truck_states if t['status'] == 'idle'),
            self.speed_multiplier,
            len(self.plant_queue)
        ], dtype=np.float32)

        # Plant Controller: Fokus pada antrean dan sisa waktu
        obs_plant = np.array([
            len(self.plant_queue),
            sum(self.plant_queue),
            (self.max_time - self.time) / self.max_time
        ], dtype=np.float32)

        return {
            "scheduler": obs_scheduler,
            "dispatcher": obs_dispatcher,
            "plant": obs_plant
        }
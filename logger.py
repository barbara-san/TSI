from stable_baselines3.common.callbacks import BaseCallback

class CustomLogger(BaseCallback):
    def __init__(self, verbose=0, model_type='DQN'):
        super(CustomLogger, self).__init__(verbose)
        self.verbose = verbose
        self.headway = {}
        self.speeds = {}
        self.crashed = {}
        self.env = None
        self.reset = False
        self.steps = 0.0
        self.ep = 1
        self.model_type = model_type
        self.lane_changes = 0

    def _on_training_start(self):
        self.env = self.training_env.envs[0].env.original_env
        self.headway = {f'{i}':0 for i in range(len(self.env.controlled_vehicles))}
        self.speeds = {f'{i}':0 for i in range(len(self.env.controlled_vehicles))}
        self.crashed = {f'{i}':False for i in range(len(self.env.controlled_vehicles))}

    def _on_step(self):

        dones = self.locals.get('dones')
        infos = self.locals.get('infos')

        if any(dones) or (infos and any(info.get('terminal_observation') is not None for info in infos)):
            total_crashes = 0
            for vehicle, avg in self.headway.items():
                avg = float(avg) / self.steps
                self.logger.record(f'Avg_{vehicle}_headway', avg)
            for vehicle, avg in self.speeds.items():
                avg = float(avg) / self.steps
                self.logger.record(f'Avg_{vehicle}_speed', avg)
            for vehicle, crash in self.crashed.items():
                total_crashes += int(crash)
                self.logger.record(f'Total_Crashed_Cras', total_crashes)
            
            self.logger.record('Total_Lane_Changes', self.lane_changes)

            if self.model_type == 'DQN':
                self.model._dump_logs()
            elif self.model_type == 'PPO':
                self.model._dump_logs(self.ep)
            else:
                raise TypeError('Girl, Invalid Model Type')
            self.ep += 1
            self.reset = not self.reset

        if self.reset:
            self.steps = 0.0
            self.lane_changes = 0
            self.crashed = {f'{i}':False for i in range(len(self.env.controlled_vehicles))}
            self.headway = {f'{i}':0 for i in range(len(self.env.controlled_vehicles))}
            self.speeds = {f'{i}':0 for i in range(len(self.env.controlled_vehicles))}

        i = 0
        self.steps += 1

        controlled_vehicles_x = [vehicle.to_dict()["x"] for vehicle in self.env.controlled_vehicles]   

        for vehicle in self.env.controlled_vehicles:
            self.speeds[f'{i}'] += vehicle.speed
            self.crashed[f'{i}'] = vehicle.crashed
            self.lane_changes += float(vehicle.lane_offset[2] != 0)

            vehicles_in_front_x = list(filter(lambda x: x > vehicle.to_dict()["x"], controlled_vehicles_x))
            headway_distance = (vehicles_in_front_x[0] - vehicle.to_dict()["x"]) if len(vehicles_in_front_x) > 0 else -1
            if headway_distance == -1:
                i += 1
                continue
            else:
                self.headway[f'{i}'] += headway_distance
            i += 1

        return True
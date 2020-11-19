class ExplorationRate:
    def __init__(self, eps):
        self.eps_start = eps[0]
        self.eps_end = eps[1]
        self.eps_steps = eps[2]

    def get_rate(self, episode_num):
        if episode_num > self.eps_steps:
            return self.eps_end

        return (self.eps_end - self.eps_start) / self.eps_steps * episode_num + self.eps_start

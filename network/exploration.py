class ExplorationRate:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

    def get_rate(self, episode_num):
        if episode_num > self.eps_end:
            return self.eps_end

        return (self.eps_end - self.eps_start) / self.eps_steps * episode_num + self.eps_start

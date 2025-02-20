import torch

import cubesim
import cubesim.visualizer
import network.models
import network.replay


class Agent:
    def __init__(self, replay_size, reward_fn, device, nn_params):
        self.memory = network.replay.ReplayMemory(replay_size)
        self.reward_fn = reward_fn

        cube = cubesim.Cube2()
        self.target_net = network.models.DQN(cube.embedding_dim[0] * cube.embedding_dim[1],
                                             len(cube.moves), **nn_params).to(device)
        self.policy_net = network.models.DQN(cube.embedding_dim[0] * cube.embedding_dim[1],
                                             len(cube.moves), **nn_params).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        with torch.no_grad():
            qvals = self.policy_net(state)
        return qvals.argmax()

    def optimize_model(self, optimizer, criterion, batch_size, gamma):
        if len(self.memory) < batch_size:
            return 0

        samples = self.memory.sample(batch_size)  # this is in AoS format
        batch = network.replay.Transition(*zip(*samples))  # this is in SofA format

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_batch = torch.cat(batch.next_state)

        # Run the recorded next states through the model. This return the qvalues for
        # each action. We take the maximum qval from each state. We stop tracking
        # gradients since this ends up being manipulated into the target.
        with torch.no_grad():
            next_state_qvals = self.target_net(next_batch).max(1)[0]
            target_actions_qvals = (next_state_qvals * gamma) + reward_batch

        self.policy_net.train()
        # Run the recorded states through the model. This returns the qvalues for
        # each action. We use `gather` to select the qvalues of the previously chosen
        # action. This corresponds to Q(s_j, a_j).
        output = self.policy_net(state_batch)
        generated_actions_qvals = output.gather(1, action_batch.unsqueeze(1)).squeeze()

        loss = criterion(generated_actions_qvals, target_actions_qvals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss)

    def play_episode(self, optimizer, criterion, scramble, batch_size, gamma, epsilon, num_steps, device, layer_mode=False):
        """The agent attempts to solve the scramble provided in `num_steps` moves.
        Returns:
            list: A list countaining the losses through `num_steps` moves.
        """

        cube = cubesim.Cube2()
        cube.load_scramble(scramble)
        state = cube.get_embedding(device).unsqueeze(0)

        if (layer_mode and cube.layer_solved()):
            return [0.0]

        if (not layer_mode and cube.is_solved()):
            return [0.0]

        history = []

        randoms = torch.rand(num_steps)
        for i in range(num_steps):
            if randoms[i] < epsilon:
                action = torch.randint(12, (1,), device=device)
                cube.moves[action]()
            else:
                action = self.select_action(state).view(1)
                cube.moves[action]()

            next_state = cube.get_embedding(device).unsqueeze(0)
            reward = self.reward_fn(next_state)

            self.memory.push(state, action, next_state, torch.tensor([reward], device=device))
            state = next_state

            loss = self.optimize_model(optimizer, criterion, batch_size, gamma)
            history.append(loss)

            if reward > 0:
                break

        return history

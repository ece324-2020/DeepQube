import cubesim
import network.models
import network.replay

class Agent:
    def __init__(self, replay_size, reward_fn, device, nn_params):
        self.memory = network.replay.ReplayMemory(replay_size)
        self.reward_fn = reward_fn

        cube = cubesim.Cube2()
        self.model = network.models.DQN(cube.embedding_dim,
                len(cube.moves), **nn_params).to(device)

    def select_action(self, state):
        qvals = self.model(state)
        return int(qvals.argmax())

    def optimize_model(self, optimizer, criterion, batch_size, gamma):
        if len(self.memory) < batch_size:
            return 0

        samples = self.memory.sample(batch_size) # this is in AoS format
        batch = network.replay.Transition(*zip(*samples)) # this is in SofA format

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_batch = torch.cat(batch.next_state)

        self.model.eval()
        # Run the recorded next states through the model. This return the qvalues for
        # each action. We take the maximum qval from each state. We stop tracking
        # gradients since this ends up being manipulated into the target.
        with torch.no_grad():
            next_state_qvals = self.model(next_batch).max(1)[0]
            target_actions_qvals = (next_state_qvals * gamma) + reward_batch

        self.model.train()
        # Run the recorded states through the model. This returns the qvalues for
        # each action. We use `gather` to select the qvalues of the previously chosen
        # action. This corresponds to Q(s_j, a_j).
        generated_actions_qvals = self.model(state_batch).gather(1, action_batch)

        loss = criterion(next_state_qvals, target_actions_qvals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss)

        
    def play_episode(self, optimizer, criterion, scramble, batch_size, gamma, num_steps):
        """The agent attempts to solve the scramble provided in `num_steps` moves.
        Returns:
            list: A list countaining the losses through `num_steps` moves.
        """

        cube = cubesim.Cube2()
        cube.load_scramble(scramble)
        state = cube.get_embedding(self.model.device)

        history = []

        for i in range(num_steps):
            action = self.select_action(state)
            cube.moves[action]()

            next_state = cube.get_embedding(self.model.device)
            reward = self.reward_fn(next_state)

            self.memory.push(state, action, next_state, reward)
            state = next_state

            loss = self.optimize_model(optimizer, criterion, batch_size, gamma)
            history.append(loss)
            # TODO: stop if we solved the cube

        return history


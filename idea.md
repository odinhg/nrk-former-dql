n_episodes = 0

START_EPISODE:

Initalize board to state S

PICK_ACTION:

Choose valid action A based on epsilon-greedy strategy:
	with probability epsilon pick a random action
	with probability 1 - epsilon pick the action with the highest Q-value from the main network

Perform action A
Get reward R = reward(S, A)
Observe new state S'
T = is S' terminal?

Store (S,A,R,S',T) into replay memory

Compute loss value based on reward and target network output
Optimize main network

IF T:
	n_episodes += 1
	REPLAY
	START_EPISODE
	if (n_episodes + 1) % 100 == 0:
		update target network: copy weights from main network to target network
ELSE:
	PICK_ACTION



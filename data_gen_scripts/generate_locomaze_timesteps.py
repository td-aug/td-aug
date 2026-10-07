import glob
import json
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from agents import SACAgent
from tqdm import tqdm
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'navigate', 'Dataset type.')
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_timesteps', 1_000_000, 'Number of timesteps.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode, will terminate if exceed')
flags.DEFINE_integer('min_episode_steps', 10, 'Minimum number of steps in an episode, will not accept if less')
flags.DEFINE_boolean('only_success', False, 'Only keep episodes that terminated by reaching the goal.')
# flags.DEFINE_boolean('terminate', False, 'Terminate at goal.')

def main(_):
    assert FLAGS.dataset_type in ['path', 'navigate', 'stitch', 'explore']
    # 'path': Reach a single goal and stay there.
    # 'navigate': Repeatedly reach randomly sampled goals in a single episode.
    # 'stitch': Reach a nearby goal that is 4 cells away and stay there.
    # 'explore': Repeatedly follow random directions sampled every 10 steps.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        # terminate_at_goal=False,
        # terminate_at_goal=FLAGS.terminate,
        terminate_at_goal=True,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    ob_dim = env.observation_space.shape[0]

    # Initialize oracle agent.
    if 'point' in FLAGS.env_name:

        def actor_fn(ob, temperature):
            return ob[-2:]
    else:
        # Load agent config.
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

        with open(candidates[0] + '/flags.json', 'r') as f:
            agent_config = json.load(f)['agent']

        # Load agent.
        agent = SACAgent.create(
            FLAGS.seed,
            np.zeros(ob_dim),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    # Store all empty cells and vertex cells.
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

                # Exclude hallway cells.
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    # num_train_episodes = FLAGS.num_episodes
    # num_val_episodes = FLAGS.num_episodes // 10
    min_num_train_steps = FLAGS.num_timesteps
    min_num_val_timesteps = FLAGS.num_timesteps // 10
    final_total_steps = min_num_train_steps + min_num_val_timesteps
    # for ep_idx in trange(num_train_episodes + num_val_episodes):
    # while True:
    # while total_steps < final_total_steps:

    with tqdm(total=final_total_steps) as pbar:
        while not (total_train_steps >= min_num_train_steps and total_steps - total_train_steps >= min_num_val_timesteps):
            episode_dataset = defaultdict(list)
            if FLAGS.dataset_type in ['path', 'navigate', 'explore']:
                # Sample an initial state from all cells.
                init_ij = all_cells[np.random.randint(len(all_cells))]
                # Sample a goal state from vertex cells.
                goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
            elif FLAGS.dataset_type == 'stitch':
                # Sample an initial state from all cells.
                init_ij = all_cells[np.random.randint(len(all_cells))]

                # Perform BFS to find adjacent cells.
                adj_cells = []
                adj_steps = 4  # Target distance from the initial cell.
                bfs_map = maze_map.copy()
                for i in range(bfs_map.shape[0]):
                    for j in range(bfs_map.shape[1]):
                        bfs_map[i][j] = -1
                bfs_map[init_ij[0], init_ij[1]] = 0
                queue = [init_ij]
                while len(queue) > 0:
                    i, j = queue.pop(0)
                    for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < bfs_map.shape[0]
                            and 0 <= nj < bfs_map.shape[1]
                            and maze_map[ni, nj] == 0
                            and bfs_map[ni, nj] == -1
                        ):
                            bfs_map[ni][nj] = bfs_map[i][j] + 1
                            queue.append((ni, nj))
                            if bfs_map[ni][nj] == adj_steps:
                                adj_cells.append((ni, nj))

                # Sample a goal state from adjacent cells.
                goal_ij = adj_cells[np.random.randint(len(adj_cells))] if len(adj_cells) > 0 else init_ij
            else:
                raise ValueError(f'Unsupported dataset_type: {FLAGS.dataset_type}')
            # task = [(1, 1), (10, 14)]
            # task = [(8, 3), (5, 12)]
            # tasks = [[(1, 1), (10, 14)],
            #         [(1, 14), (10, 1)],
            #         [(8, 14), (1, 1)],
            #         [(8, 3), (5, 12)],
            #         [(5, 9), (3, 8)],]
            # task = tasks[np.random.randint(len(tasks))]

            ob, _ = env.reset(options=dict(task_info=dict(init_ij=init_ij, goal_ij=goal_ij)))

            done = False
            step = 0

            cur_subgoal_dir = None  # Current subgoal direction (only for 'explore').

            while not done:
                if FLAGS.dataset_type == 'explore':
                    # Sample a random direction every 10 steps.
                    if step % 10 == 0:
                        cur_subgoal_dir = np.random.randn(2)
                        cur_subgoal_dir = cur_subgoal_dir / (np.linalg.norm(cur_subgoal_dir) + 1e-6)
                    subgoal_dir = cur_subgoal_dir
                else:
                    # Get the oracle subgoal and compute the direction.
                    subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
                    subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
                    subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

                agent_ob = env.unwrapped.get_ob(ob_type='states')
                # Exclude the agent's position and add the subgoal direction.
                agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
                action = actor_fn(agent_ob, temperature=0)
                # Add Gaussian noise to the action.
                action = action + np.random.normal(0, FLAGS.noise, action.shape)
                action = np.clip(action, -1, 1)

                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                success = info['success']

                # Sample a new goal state when the current goal is reached.
                if success and FLAGS.dataset_type == 'navigate':
                    goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
                    env.unwrapped.set_goal(goal_ij)

                episode_dataset['observations'].append(ob)
                episode_dataset['actions'].append(action)
                episode_dataset['terminals'].append(done)
                episode_dataset['qpos'].append(info['prev_qpos'])
                episode_dataset['qvel'].append(info['prev_qvel'])
                episode_dataset['success'].append(success)

                ob = next_ob
                step += 1

            episode_success = len(episode_dataset['success']) > 0 and episode_dataset['success'][-1] == 1.0

            # Add to actual dataset if episode length is large enough. By default,
            # keep only trajectories that ended by reaching the goal, so terminal
            # states are native goal terminals rather than time-limit truncations.
            if step > FLAGS.min_episode_steps and ((not FLAGS.only_success) or episode_success):
                for k, v in episode_dataset.items():
                    dataset[k].extend(v)

                # if ep_idx < num_train_episodes:
                if total_steps < min_num_train_steps:
                    total_train_steps += step
                total_steps += step
                pbar.update(step)

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    terminals = np.array(dataset['terminals'])
    ends = np.arange(terminals.shape[0])[terminals]
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts
    success = np.array(dataset['success'])
    success_rate = success.sum() / terminals.sum()
    print('Success rate:', success_rate, 'over ', len(ends), 'episodes')
    # mean length of succesful trajectories
    print('mean length of traj', lengths.mean())
    print('Mean length of succesful traj', lengths[success[terminals] == 1].mean())
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)
    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)

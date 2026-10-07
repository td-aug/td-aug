from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Number of episodes.')
flags.DEFINE_bool(
    'save_subtask_metadata',
    False,
    'Save the active data-collection target and oracle-plan boundaries.',
)


SUBTASK_BOOL_KEYS = {
    'subtask_terminals',
    'subtask_successes',
    'target_is_stack',
}
SUBTASK_INT_KEYS = {
    'episode_ids',
    'subtask_ids',
    'subtask_indices',
    'target_tasks',
    'target_blocks',
    'target_buttons',
    'target_button_states',
}

SCENE_TARGET_TASK_IDS = {
    'cube': 0,
    'button': 1,
    'drawer': 2,
    'window': 3,
}


def main(_):
    assert FLAGS.dataset_type in ['play', 'noisy']
    if FLAGS.save_subtask_metadata and not any(
        env_type in FLAGS.env_name for env_type in ('cube', 'puzzle', 'scene')
    ):
        raise ValueError(
            'Subtask metadata is currently supported only for cube, puzzle, '
            'and scene environments.'
        )

    np.random.seed(FLAGS.seed)
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )

    # Initialize oracles.
    oracle_type = 'plan' if FLAGS.dataset_type == 'play' else 'markov'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    if 'cube' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'scene' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'window': WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'button': ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'drawer': DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'window': WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'puzzle' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
            }
        else:
            agents = {
                'button': ButtonPlanOracle(
                    env=env,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    next_subtask_id = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        while True:
            reset_seed = FLAGS.seed if ep_idx == 0 else None
            ob, info = env.reset(seed=reset_seed)

            # Set the cube stacking probability for this episode.
            if 'single' in FLAGS.env_name:
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif 'quadruple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            else:
                p_stack = 0.5

            if oracle_type == 'markov':
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info['privileged/target_task']]
            agent.reset(ob, info)

            done = False
            step = 0
            ep_qpos = []
            subtask_id = next_subtask_id
            next_subtask_id += 1
            subtask_index = 0

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == 'markov':
                        # Add Gaussian noise to the action.
                        action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Capture metadata before set_new_target replaces the active target.
                subtask_terminal = bool(agent.done)
                if FLAGS.save_subtask_metadata:
                    subtask_success = bool(info['success'])
                    if 'cube' in FLAGS.env_name:
                        target_block = int(info['privileged/target_block'])
                        target_block_pos = np.asarray(
                            info['privileged/target_block_pos'], dtype=np.float32
                        ).copy()
                        target_block_yaw = float(
                            info['privileged/target_block_yaw'][0]
                        )
                    elif 'puzzle' in FLAGS.env_name:
                        target_button = int(info['privileged/target_button'])
                        target_button_state = int(
                            info['privileged/target_button_state']
                        )
                        target_button_pos = np.asarray(
                            info['privileged/target_button_top_pos'], dtype=np.float32
                        ).copy()
                    elif 'scene' in FLAGS.env_name:
                        target_task = SCENE_TARGET_TASK_IDS[
                            info['privileged/target_task']
                        ]
                        target_block = int(info['privileged/target_block'])
                        target_block_pos = np.asarray(
                            info['privileged/target_block_pos'], dtype=np.float32
                        ).copy()
                        target_block_yaw = float(
                            info['privileged/target_block_yaw'][0]
                        )
                        target_button = int(info['privileged/target_button'])
                        target_button_state = int(
                            info['privileged/target_button_state']
                        )
                        target_button_pos = np.asarray(
                            info['privileged/target_button_top_pos'], dtype=np.float32
                        ).copy()
                        target_drawer_pos = float(
                            info['privileged/target_drawer_pos'][0]
                        )
                        target_drawer_handle_pos = np.asarray(
                            info['privileged/target_drawer_handle_pos'],
                            dtype=np.float32,
                        ).copy()
                        target_window_pos = float(
                            info['privileged/target_window_pos'][0]
                        )
                        target_window_handle_pos = np.asarray(
                            info['privileged/target_window_handle_pos'],
                            dtype=np.float32,
                        ).copy()

                if subtask_terminal:
                    # Set a new task when the current task is done.
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info['privileged/target_task']]
                    agent.reset(agent_ob, agent_info)

                dataset['observations'].append(ob)
                dataset['actions'].append(action)
                dataset['terminals'].append(done)
                dataset['qpos'].append(info['prev_qpos'])
                dataset['qvel'].append(info['prev_qvel'])
                if FLAGS.save_subtask_metadata:
                    dataset['episode_ids'].append(ep_idx)
                    dataset['subtask_ids'].append(subtask_id)
                    dataset['subtask_indices'].append(subtask_index)
                    dataset['subtask_terminals'].append(subtask_terminal)
                    dataset['subtask_successes'].append(subtask_success)
                    if 'cube' in FLAGS.env_name:
                        dataset['target_blocks'].append(target_block)
                        dataset['target_block_positions'].append(target_block_pos)
                        dataset['target_block_yaws'].append(target_block_yaw)
                        dataset['target_is_stack'].append(target_block_pos[2] > 0.04)
                    elif 'puzzle' in FLAGS.env_name:
                        dataset['target_buttons'].append(target_button)
                        dataset['target_button_states'].append(target_button_state)
                        dataset['target_button_positions'].append(target_button_pos)
                    elif 'scene' in FLAGS.env_name:
                        dataset['target_tasks'].append(target_task)
                        dataset['target_blocks'].append(target_block)
                        dataset['target_block_positions'].append(target_block_pos)
                        dataset['target_block_yaws'].append(target_block_yaw)
                        dataset['target_buttons'].append(target_button)
                        dataset['target_button_states'].append(target_button_state)
                        dataset['target_button_positions'].append(target_button_pos)
                        dataset['target_drawer_positions'].append(target_drawer_pos)
                        dataset['target_drawer_handle_positions'].append(
                            target_drawer_handle_pos
                        )
                        dataset['target_window_positions'].append(target_window_pos)
                        dataset['target_window_handle_positions'].append(
                            target_window_handle_pos
                        )
                if has_button_states:
                    dataset['button_states'].append(info['prev_button_states'])
                ep_qpos.append(info['prev_qpos'])

                if subtask_terminal and not done:
                    subtask_id = next_subtask_id
                    next_subtask_id += 1
                    subtask_index += 1

                ob = next_ob
                step += 1

            if 'scene' in FLAGS.env_name:
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False  # Block goes too far left, without being in the drawer.

                if is_healthy:
                    break
                else:
                    # Remove the last episode and retry.
                    print('Unhealthy episode, retrying...', flush=True)
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-step]
            else:
                break

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals' or k in SUBTASK_BOOL_KEYS:
            dtype = bool
        elif k == 'button_states':
            dtype = np.int64
        elif k in SUBTASK_INT_KEYS:
            dtype = np.int32
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    if FLAGS.save_subtask_metadata:
        num_subtasks = len(np.unique(train_dataset['subtask_ids']))
        num_completed = int(np.sum(train_dataset['subtask_terminals']))
        num_successful = int(
            np.sum(
                train_dataset['subtask_terminals']
                & train_dataset['subtask_successes']
            )
        )
        print('Total train subtasks:', num_subtasks)
        print('Completed train subtasks:', num_completed)
        print('Successful completed train subtasks:', num_successful)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)

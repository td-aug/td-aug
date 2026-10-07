import argparse

import numpy as np


COMMON_REQUIRED_KEYS = {
    'observations',
    'actions',
    'terminals',
    'qpos',
    'qvel',
    'episode_ids',
    'subtask_ids',
    'subtask_indices',
    'subtask_terminals',
    'subtask_successes',
}
CUBE_REQUIRED_KEYS = {
    'target_blocks',
    'target_block_positions',
    'target_block_yaws',
    'target_is_stack',
}
PUZZLE_REQUIRED_KEYS = {
    'button_states',
    'target_buttons',
    'target_button_states',
    'target_button_positions',
}
SCENE_REQUIRED_KEYS = {
    'button_states',
    'target_tasks',
    'target_blocks',
    'target_block_positions',
    'target_block_yaws',
    'target_buttons',
    'target_button_states',
    'target_button_positions',
    'target_drawer_positions',
    'target_drawer_handle_positions',
    'target_window_positions',
    'target_window_handle_positions',
}


def validate(path):
    with np.load(path) as dataset:
        dataset_keys = set(dataset.files)
        if 'target_tasks' in dataset_keys:
            required_keys = COMMON_REQUIRED_KEYS | SCENE_REQUIRED_KEYS
            target_type = 'scene'
        elif 'target_blocks' in dataset_keys:
            required_keys = COMMON_REQUIRED_KEYS | CUBE_REQUIRED_KEYS
            target_type = 'cube'
        elif 'target_buttons' in dataset_keys:
            required_keys = COMMON_REQUIRED_KEYS | PUZZLE_REQUIRED_KEYS
            target_type = 'puzzle'
        else:
            raise ValueError(
                'Dataset has neither cube, puzzle, nor scene target metadata.'
            )

        missing = required_keys - dataset_keys
        if missing:
            raise ValueError(f'Missing required keys: {sorted(missing)}')

        lengths = {key: len(dataset[key]) for key in dataset.files}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'Array lengths are not aligned: {lengths}')

        size = lengths['actions']
        if size == 0:
            raise ValueError('Dataset is empty.')

        terminals = np.asarray(dataset['terminals'], dtype=bool)
        switches = np.asarray(dataset['subtask_terminals'], dtype=bool)
        subtask_ids = np.asarray(dataset['subtask_ids'])
        episode_ids = np.asarray(dataset['episode_ids'])

        id_changes = np.flatnonzero(np.diff(subtask_ids) != 0)
        expected_changes = np.flatnonzero((terminals | switches)[:-1])
        if not np.array_equal(id_changes, expected_changes):
            raise ValueError(
                'Subtask IDs do not change exactly after subtask switches or '
                'episode terminals.'
            )

        episode_changes = np.flatnonzero(np.diff(episode_ids) != 0)
        expected_episode_changes = np.flatnonzero(terminals[:-1])
        if not np.array_equal(episode_changes, expected_episode_changes):
            raise ValueError('Episode IDs do not change exactly after terminals.')

        same_subtask = np.diff(subtask_ids) == 0
        if target_type == 'cube':
            target_blocks = np.asarray(dataset['target_blocks'])
            target_positions = np.asarray(dataset['target_block_positions'])
            target_yaws = np.asarray(dataset['target_block_yaws'])
            if np.any(
                target_blocks[1:][same_subtask]
                != target_blocks[:-1][same_subtask]
            ):
                raise ValueError('Target block changes within a subtask.')
            if not np.allclose(
                target_positions[1:][same_subtask],
                target_positions[:-1][same_subtask],
            ):
                raise ValueError('Target position changes within a subtask.')
            if not np.allclose(
                target_yaws[1:][same_subtask],
                target_yaws[:-1][same_subtask],
            ):
                raise ValueError('Target yaw changes within a subtask.')
        elif target_type == 'puzzle':
            target_buttons = np.asarray(dataset['target_buttons'])
            target_states = np.asarray(dataset['target_button_states'])
            target_positions = np.asarray(dataset['target_button_positions'])
            if np.any(
                target_buttons[1:][same_subtask]
                != target_buttons[:-1][same_subtask]
            ):
                raise ValueError('Target button changes within a subtask.')
            if np.any(
                target_states[1:][same_subtask]
                != target_states[:-1][same_subtask]
            ):
                raise ValueError('Target button state changes within a subtask.')
            if not np.allclose(
                target_positions[1:][same_subtask],
                target_positions[:-1][same_subtask],
            ):
                raise ValueError('Target button position changes within a subtask.')
        else:
            target_tasks = np.asarray(dataset['target_tasks'])
            if np.any((target_tasks < 0) | (target_tasks > 3)):
                raise ValueError('Scene target task IDs must be in [0, 3].')
            if np.any(
                target_tasks[1:][same_subtask]
                != target_tasks[:-1][same_subtask]
            ):
                raise ValueError('Scene target task changes within a subtask.')

            integer_target_keys = (
                'target_blocks',
                'target_buttons',
                'target_button_states',
            )
            float_target_keys = (
                'target_block_positions',
                'target_block_yaws',
                'target_button_positions',
                'target_drawer_positions',
                'target_drawer_handle_positions',
                'target_window_positions',
                'target_window_handle_positions',
            )
            for key in integer_target_keys:
                values = np.asarray(dataset[key])
                if np.any(values[1:][same_subtask] != values[:-1][same_subtask]):
                    raise ValueError(f'{key} changes within a Scene subtask.')
            for key in float_target_keys:
                values = np.asarray(dataset[key])
                if not np.allclose(
                    values[1:][same_subtask], values[:-1][same_subtask]
                ):
                    raise ValueError(f'{key} changes within a Scene subtask.')

        completed_successes = switches & np.asarray(
            dataset['subtask_successes'], dtype=bool
        )
        print(f'Validated: {path}')
        print(f'Target type: {target_type}')
        print(f'Transitions: {size}')
        print(f'Episodes: {int(terminals.sum())}')
        print(f'Subtasks: {len(np.unique(subtask_ids))}')
        print(f'Switches: {int(switches.sum())}')
        print(f'Successful switches: {int(completed_successes.sum())}')
        if target_type == 'scene':
            task_names = ('cube', 'button', 'drawer', 'window')
            target_tasks = np.asarray(dataset['target_tasks'])
            for task_id, task_name in enumerate(task_names):
                print(
                    f'{task_name.title()} transitions: '
                    f'{int(np.sum(target_tasks == task_id))}'
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    validate(args.path)


if __name__ == '__main__':
    main()

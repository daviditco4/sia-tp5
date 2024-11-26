import subprocess
from tqdm import tqdm  # For the progress bar

# Define the command template and the arguments
command_template = 'python3'
arguments = [
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/optimizer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/learning_rate_-00005.json outputs/noisy/learning_rate.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/learning_rate_-0002.json outputs/noisy/learning_rate.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/learning_rate_-0005.json outputs/noisy/learning_rate.csv 0.1 4',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_35.json outputs/noisy/largest_layer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_50.json outputs/noisy/largest_layer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_100.json outputs/noisy/largest_layer.csv 0.1 4',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/momentum_-8.json outputs/noisy/optimizer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/momentum_-95.json outputs/noisy/optimizer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none.json outputs/noisy/optimizer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none_momentum_-8.json outputs/noisy/optimizer.csv 0.1 4',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none_momentum_-95.json outputs/noisy/optimizer.csv 0.1 4',
]

# Number of repetitions
num_repetitions = 1


def run_command_with_progress(command, args, repetitions):
    total_tasks = len(args) * repetitions  # Total number of commands to execute
    with tqdm(total=total_tasks, desc="Processing Commands") as pbar:
        for arg in args:
            for _ in range(repetitions):
                # Construct the full command
                full_command = [command] + arg.split()
                try:
                    # Execute the command
                    result = subprocess.run(full_command, capture_output=True, text=True, check=True)
                    print(f'Command executed: {result.args}')
                    print('Output:')
                    print(result.stdout.strip())
                except subprocess.CalledProcessError as e:
                    print(f'Error executing command: {e}')
                    print(f'Command: {e.cmd}')
                    print(f'Output: {e.output}')
                    print(f'Error: {e.stderr}')
                finally:
                    # Update the progress bar
                    pbar.update(1)


if __name__ == "__main__":
    run_command_with_progress(command_template, arguments, num_repetitions)
import subprocess

# Define the command template and the arguments
command_template = 'python3'
arguments = [
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-1.csv 0.1 2',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_35.json outputs/largest_layer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_50.json outputs/largest_layer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/largest_layer_100.json outputs/largest_layer.csv',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/momentum_-8.json outputs/optimizer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/momentum_-95.json outputs/optimizer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none.json outputs/optimizer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none_momentum_-8.json outputs/optimizer.csv',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/optimizer_none_momentum_-95.json outputs/optimizer.csv',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-3.csv 0.3',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-3.csv 0.3 2',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-3.csv 0.3 4',

    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-5.csv 0.5',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-5.csv 0.5 2',
    'exercise1_discriminative/task2_denoising/runner_main.py configs/prototype.json outputs/noisy/noise0-5.csv 0.5 4',
]

# Number of repetitions
num_repetitions = 1


def run_command(command, args, repetitions):
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


if __name__ == "__main__":
    run_command(command_template, arguments, num_repetitions)
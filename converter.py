from functools import partial
from multiprocessing import Pool
from pathlib import Path
from subprocess import call


def main():
    data_dir = Path('/home/darkroom2/Downloads/ModelNet10/ModelNet10/')

    configs = [
        {
            'output_dir': 'output/view_1',
            'config_file': 'config/f3d_config.json',
            'scenario_file': 'config/f3d_scenario_left.log'
        },
        {
            'output_dir': 'output/view_2',
            'config_file': 'config/f3d_config.json',
            'scenario_file': 'config/f3d_scenario_up.log'
        }
    ]

    for config in configs:
        output_dir = Path(config['output_dir'])
        config_file = Path(config['config_file'])
        scenario_file = Path(config['scenario_file'])

        with Pool() as pool:
            _run_f3d = partial(run_f3d, data_dir=data_dir, config_file=config_file, output_dir=output_dir,
                               scenario_file=scenario_file)
            pool.map(_run_f3d, data_dir.glob('**/*.off'))


def run_f3d(file, data_dir, config_file, output_dir, scenario_file):
    output_file = output_dir.joinpath(file.relative_to(data_dir)).with_suffix('.png')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'flatpak run io.github.f3d_app.f3d {file.resolve()} --config={config_file.resolve()} ' \
          f'--output={output_file.resolve()} --interaction-test-play={scenario_file.resolve()}'
    call(cmd.split(' '))


if __name__ == '__main__':
    main()

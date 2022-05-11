from functools import partial
from multiprocessing import Pool
from pathlib import Path
from subprocess import call


def modelnet_to_n_views(data_dir_path, output_dir_path, config_file_path, renderer_scenarios_paths):
    data_dir = Path(data_dir_path)
    output_dir = Path(output_dir_path)
    config_file = Path(config_file_path)
    for idx, scenario in enumerate(renderer_scenarios_paths):
        scenario_file = Path(scenario)
        with Pool() as pool:
            _run_f3d = partial(run_f3d, output_dir=output_dir, config_file=config_file, scenario_file=scenario_file,
                               name_suffix=f'_v{idx}')
            pool.map(_run_f3d, data_dir.glob('**/*.off'))


def run_f3d(file, output_dir, config_file, scenario_file, name_suffix=None):
    file_name = file.stem
    if name_suffix:
        file_name += name_suffix
    output_file = output_dir.joinpath(file_name).with_suffix('.png')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'flatpak run io.github.f3d_app.f3d {file.resolve()} --config={config_file.resolve()} ' \
          f'--output={output_file.resolve()} --interaction-test-play={scenario_file.resolve()}'
    call(cmd.split(' '))


def main():
    data_dir_path = '/home/darkroom2/Downloads/ModelNet10/ModelNet10/'
    output_dir_path = './output/'
    config_file_path = './config/f3d_config.json'
    renderer_scenarios_paths = ['./config/f3d_scenario_left.log', './config/f3d_scenario_up.log']

    modelnet_to_n_views(data_dir_path, output_dir_path, config_file_path, renderer_scenarios_paths)


if __name__ == '__main__':
    main()

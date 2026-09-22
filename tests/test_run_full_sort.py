import importlib.util
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).parents[1] / 'tools' / 'run_full_sort.py'
    spec = importlib.util.spec_from_file_location('run_full_sort', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_override_can_add_a_new_registered_setting_to_old_ops():
    runner = _load_runner()
    settings = {'fs': 20_000}

    overrides = runner.apply_overrides(
        settings, ['final_merge_borderline_rescue=true'])

    assert overrides == {'final_merge_borderline_rescue': True}
    assert settings['final_merge_borderline_rescue'] is True


def test_override_still_rejects_typos():
    runner = _load_runner()

    with pytest.raises(ValueError, match='not a registered Kilosort setting'):
        runner.apply_overrides({}, ['final_merge_borderline_resuce=true'])

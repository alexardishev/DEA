from src.utils.io_utils import load_yaml


def test_config_has_dea_section():
    cfg = load_yaml('config/pipeline_config.yaml')
    assert 'analysis' in cfg
    assert 'dea' in cfg['analysis']

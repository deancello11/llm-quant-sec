import yaml
from src.common import ROOT, load_config


def test_every_template_has_placeholder():
    t = yaml.safe_load(open(ROOT / load_config()["templates_file"], encoding="utf-8"))["templates"]
    assert len(t) >= 3
    for name, s in t.items():
        assert s.count("{behavior}") == 1, name
        assert "{" not in s.replace("{behavior}", ""), name

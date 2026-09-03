import pytest

from mlp.config import Params, validate


def _params(**overrides):
    values = {"input_var": 2, "output_var": 1}
    values.update(overrides)
    return Params(**values)


@pytest.mark.parametrize(
    "name,value",
    (
        ("activation", "swish"),
        ("norm", "instance"),
        ("input_normalization", "robust"),
        ("output_normalization", "robust"),
        ("output_activation", "linear"),
        ("loss", "relative_l2"),
    ),
)
def test_native_validation_rejects_unknown_closed_choices(name, value):
    with pytest.raises(SystemExit, match=name):
        validate(_params(**{name: value}))


def test_native_validation_accepts_every_published_default():
    validate(_params())

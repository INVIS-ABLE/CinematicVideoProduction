import pytest

pytest.importorskip("pydantic")

from local_studio.config import MAX_IMAGES, MAX_VIDEOS, MAX_VOICE_SAMPLES, native_size
from local_studio.models import GenerationRequest


def _ids(prefix, count):
    return [f"{prefix}_{i}" for i in range(count)]


def test_exact_reference_limits_are_accepted():
    request = GenerationRequest(
        project_id="p_123456789abc",
        prompt="A cinematic test",
        image_ids=_ids("image", MAX_IMAGES),
        video_ids=_ids("video", MAX_VIDEOS),
        voice_ids=_ids("voice", MAX_VOICE_SAMPLES),
    )
    assert len(request.image_ids) == 15
    assert len(request.video_ids) == 3
    assert len(request.voice_ids) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_ids", _ids("image", MAX_IMAGES + 1)),
        ("video_ids", _ids("video", MAX_VIDEOS + 1)),
        ("voice_ids", _ids("voice", MAX_VOICE_SAMPLES + 1)),
    ],
)
def test_reference_limits_are_server_enforced(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        GenerationRequest(project_id="p_123456789abc", prompt="A test", **kwargs)


def test_selected_reference_must_be_in_request():
    with pytest.raises(ValueError):
        GenerationRequest(
            project_id="p_123456789abc",
            prompt="A test",
            image_ids=["image_1"],
            opening_image_id="image_2",
        )


def test_native_sizes_are_wan_safe_and_under_area():
    for resolution, area in {"480p": 832 * 480, "720p": 1280 * 704}.items():
        for ratio in ("16:9", "9:16", "1:1", "4:5", "2.35:1"):
            width, height = native_size(resolution, ratio)
            assert width % 32 == 0
            assert height % 32 == 0
            assert width * height <= area

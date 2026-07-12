import pytest

pytest.importorskip("pydantic")

from local_studio.models import AssetDirective, AssetKind, AssetRecord, GenerationRequest
from local_studio.planner import build_prompt


def test_reference_notes_are_preserved_in_prompt():
    asset = AssetRecord(
        asset_id="a_1234567890abcdef",
        project_id="p_123456789abc",
        kind=AssetKind.image,
        original_name="hero.png",
        relative_path="assets/image/a_1234567890abcdef.png",
        media_url="/media",
    )
    request = GenerationRequest(
        project_id=asset.project_id,
        prompt="A hero walks through rain",
        image_ids=[asset.asset_id],
        asset_directives={
            asset.asset_id: AssetDirective(
                role="character",
                note="same short silver hair and weathered red coat",
            )
        },
    )
    prompt, manifest = build_prompt(request, [asset])
    assert "same short silver hair and weathered red coat" in prompt
    assert manifest[0]["role"] == "character"

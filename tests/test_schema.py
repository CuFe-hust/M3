from data.schema import CanonicalPrediction, CanonicalSample


def test_canonical_sample_serializes_image_paths() -> None:
    sample = CanonicalSample(
        id="sample-1",
        task_type="vqa",
        images=["image.png"],
        prompt="How many ships are visible?",
        answers=["3"],
    )

    sample.validate()

    assert sample.serializable()["images"] == ["image.png"]


def test_canonical_prediction_requires_supported_task_type() -> None:
    prediction = CanonicalPrediction(id="prediction-1", task_type="caption", text="A harbor scene.")

    prediction.validate()

    assert prediction.serializable()["text"] == "A harbor scene."

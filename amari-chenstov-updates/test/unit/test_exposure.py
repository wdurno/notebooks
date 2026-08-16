import pytest

from src.exposure import stream_exposure_rows


def test_stream_exposure_rows_counts_repeated_observations_exactly() -> None:
    rows = stream_exposure_rows(
        ((1, 2, 2), (2, 3, 4), (5, 6, 7)),
        ((9, 0, 0), (0, 9, 1), (9, 9, 9)),
    )

    assert rows[0]["before_cumulative_observations"] == 0
    assert rows[0]["after_cumulative_observations"] == 3
    assert rows[0]["after_cumulative_nine_observations"] == 1
    assert rows[0]["after_cumulative_unique_observations"] == 2
    assert rows[1]["after_cumulative_observations"] == 6
    assert rows[1]["after_cumulative_nine_observations"] == 2
    assert rows[1]["after_cumulative_unique_observations"] == 4
    assert rows[1]["after_cumulative_unique_nine_observations"] == 2
    assert rows[1]["after_cumulative_unique_non_nine_observations"] == 2

    assert rows[2]["batch_consumed_by_optimizer"] is False
    assert rows[2]["batch_nine_count"] == 3
    assert rows[2]["after_cumulative_observations"] == 6
    assert rows[2]["after_cumulative_unique_observations"] == 4


def test_stream_exposure_rows_can_consume_the_final_batch() -> None:
    rows = stream_exposure_rows(
        ((1,), (2,)),
        ((0,), (9,)),
        consume_final_batch=True,
    )

    assert rows[-1]["batch_consumed_by_optimizer"] is True
    assert rows[-1]["after_cumulative_observations"] == 2
    assert rows[-1]["after_cumulative_nine_observations"] == 1


@pytest.mark.parametrize(
    "indices, labels, message",
    [
        (((1,),), ((0,), (1,)), "step counts"),
        (((1, 2),), ((0,),), "counts differ"),
        (((),), ((),), "empty"),
        (((1,),), ((10,),), "invalid MNIST label"),
    ],
)
def test_stream_exposure_rows_rejects_invalid_streams(
    indices: tuple,
    labels: tuple,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stream_exposure_rows(indices, labels)

"""A partial development bracket may propose a recipe, never select it."""

from copy import deepcopy
import pytest

from mnist_experiment.combined_memory.config import Recipe
from mnist_experiment.combined_memory.retention_match.tools.bracket_review import paired_bracket


def records():
    return [dict(unit_id=str(i), status="completed",
        metadata=dict(cohort="development", replica=0, pairing_asset_hash="paired",
            recipe=recipe.mapping()), summary=dict(Havg=havg,H100=hend))
        for i,(recipe,havg,hend) in enumerate((
            (Recipe("E",.1,.5,1e-5),2.48,4.58),
            (Recipe("E",1.,.5,1e-5),2.33,4.50),
            (Recipe("A",10.),2.38,4.56)))]


def test_paired_single_member_bracket_proposes_geometric_midpoint_without_claiming_a_result():
    rows=records(); before=deepcopy(rows)
    result=paired_bracket(rows,arm="E",member=0,low=.1,high=1.,strength=.5)
    assert result["recipe"]["ridge"] == pytest.approx(.1**.5)
    assert len(result["paired_evidence"]) == 3
    assert "proposal only" in result["reason"]
    assert rows == before


def test_binding_statistic_can_bracket_while_endpoint_already_lies_in_the_band():
    rows=records()
    rows[0]["summary"]["H100"]=4.54
    result=paired_bracket(rows,arm="E",member=0,low=.1,high=1.,strength=.5)
    assert result["bracketed_statistics"] == ["Havg"]
    assert result["already_in_band_statistics"] == ["H100"]


@pytest.mark.parametrize("defect", ["fresh", "unpaired", "unbracketed", "missing"])
def test_invalid_or_confirmation_evidence_cannot_supply_the_review(defect):
    rows=records()
    if defect=="fresh": rows[0]["metadata"]["cohort"]="fresh"
    if defect=="unpaired": rows[0]["metadata"]["pairing_asset_hash"]="different"
    if defect=="unbracketed": rows[2]["summary"]["Havg"]=3.
    if defect=="missing": rows.pop()
    with pytest.raises(ValueError):
        paired_bracket(rows,arm="E",member=0,low=.1,high=1.,strength=.5)

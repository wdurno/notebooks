"""Bounded Fix 6 unit diagnostics: no models, training, collection or hardware."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math

import pytest

from numerical_experiment.picar_coaching.coach import (
    CoachConfig, CoachingWindows, GeometricCoach, SIMULATOR_GEOMETRY,
    bicycle_step, chassis_yaw, guidance_probability, guidance_schedule,
)
from src.robotics.protocol import CommandReceipt, Control, RobotCommand
from src.robotics.simulator import PyBulletBackend, SimulatorConfig


CFG = SimulatorConfig(realtime=False, width=16, height=12, domain_override="A")


def diagnostic(*, now=0., position=(.5, 0.), yaw=0., target=(2.9, 0.),
               obstacle=(1.575, 0.), speed=0.):
    return {"privileged": True, "layout": {"start": (.5, 0.), "target": target,
            "obstacle": obstacle, "mirror": 1}, "position_m": (*position, .045),
            "orientation_xyzw": (0., 0., math.sin(yaw/2), math.cos(yaw/2)),
            "geometry": {"speed_mps": speed, "clearance_m": 2.},
            "steering_gain": 1., "simulator_time_s": now, "end_reason": None}


class FixedCoach(GeometricCoach):
    def __init__(self):
        super().__init__()
        self.proposal = {"text": "Steer left toward the ball while driving forward.",
                         "reason": "private geometry", "maneuver": "steer_left", "safety": False,
                         "diagnostics": {"position_m": [.5, 0.], "route_side": 1}}

    def propose(self, diagnostics, config):
        return deepcopy(self.proposal)


def enabled(**kwargs):
    windows = CoachingWindows(1, coach=FixedCoach(), **kwargs)
    result = windows.advance(0., 0., diagnostic(), CFG)
    assert result["text"] is not None
    return windows, result


def receipt(time, sequence=1, **kwargs):
    return CommandReceipt("unit", "session", sequence, "frame", time,
                          kwargs.pop("controls", Control(drive_mps=.1, steering_rad=-.2)),
                          kwargs.pop("accepted", True), kwargs.pop("status", "executing"), time+.5)


@pytest.mark.parametrize("n,d,p", [(0, 0., .5), (128, 0., .5), (512, .5, None),
                                  (768, None, 0.), (896, 1., 0.), (1024, 1., 0.),
                                  (1400, 1., 0.)])
def test_acquisition_schedule_exact_knots(n, d, p):
    row = guidance_schedule(n)
    if d is not None:
        assert row["difficulty"] == pytest.approx(d)
        assert row["barrier_sigma_m"] == pytest.approx(.5**d)
    if p is not None:
        assert row["coaching_probability"] == pytest.approx(p)
    assert guidance_schedule(n, condition="U")["coaching_probability"] == 0.
    assert guidance_schedule(n, initialization=True)["barrier_sigma_m"] == 1.
    assert guidance_probability(n, initialization=True) == .5


def test_schedule_smooth_monotone_and_named_rng_independent_of_global_randomness():
    rows = [guidance_schedule(n) for n in range(1025)]
    assert all(a["difficulty"] <= b["difficulty"] for a, b in zip(rows, rows[1:]))
    assert all(a["coaching_probability"] >= b["coaching_probability"] for a, b in zip(rows, rows[1:]))
    a, _ = enabled()
    b, _ = enabled()
    assert a.seed_manifest == b.seed_manifest
    assert a.seed_manifest["rng_name"] == "plan19-fix6/coach-gates"
    other = CoachingWindows(1, rng_name="plan19-fix6/other")
    assert other.seed_manifest["derived_seed"] != a.seed_manifest["derived_seed"]


@pytest.mark.parametrize("n,total", [(-1, 1024), (math.nan, 1024), (0, 0), (0, True)])
def test_invalid_clocks_fail(n, total):
    with pytest.raises(ValueError):
        guidance_schedule(n, total)


def test_camera_pan_is_not_chassis_alignment_and_turn_signs_are_explicit():
    cfg = replace(CFG, obstacle_enabled=False)
    d = diagnostic(target=(2.8, .8))
    d["executed_controls"] = asdict(Control(pan_rad=math.atan2(.8, 2.3)))
    proposal = GeometricCoach().propose(d, cfg)
    assert proposal["maneuver"] == "steer_left"
    assert proposal["diagnostics"]["sweeps"]["left"]["planned_controls"]["steering_rad"] < 0
    assert not proposal["diagnostics"]["camera_pan_used_for_alignment"]
    d = diagnostic(target=(2.8, -.8))
    assert GeometricCoach().propose(d, cfg)["maneuver"] == "steer_right"


def test_neutral_band_and_persistent_heading_hysteresis():
    cfg = replace(CFG, obstacle_enabled=False)
    coach = GeometricCoach()
    maneuvers = [coach.propose(diagnostic(now=i, yaw=-math.radians(error)), cfg)["maneuver"]
                 for i, error in enumerate((16, 9, 4, 9, 13))]
    assert maneuvers == ["steer_left", "steer_left", "forward", "forward", "steer_left"]


def test_both_passage_sides_and_persistence_until_entire_footprint_passes():
    coach = GeometricCoach()
    first = coach.propose(diagnostic(), CFG)
    assert first["diagnostics"]["route_side"] == 1
    # Chassis-center drift toward the other side must not flip the committed side.
    second = coach.propose(diagnostic(now=.1, position=(.6, -.02)), CFG)
    assert second["diagnostics"]["route_side"] == 1
    partial = coach.propose(diagnostic(now=10., position=(1.75, .6)), CFG)
    assert partial["diagnostics"]["route_side"] == 1
    assert not partial["diagnostics"]["full_footprint_passed_barrier"]
    cleared = coach.propose(diagnostic(now=20., position=(1.95, .6)), CFG)
    assert cleared["diagnostics"]["route_side"] is None
    assert cleared["diagnostics"]["full_footprint_passed_barrier"]
    right = GeometricCoach().propose(diagnostic(obstacle=(1.575, .2)), CFG)
    assert right["diagnostics"]["route_side"] == -1
    wall_side = GeometricCoach().propose(diagnostic(obstacle=(1.575, .6), target=(2.9, .8)), CFG)
    assert wall_side["diagnostics"]["passage_feasible"]["1"] is False
    assert wall_side["diagnostics"]["route_side"] == -1


def test_actual_calibration_latency_and_full_wheel_width_affect_sweeps():
    d = diagnostic(position=(3., 0.), speed=.15)
    d["control_latency_s"] = 1.5
    cfg = replace(CFG, obstacle_enabled=False,
                  calibration=replace(CFG.calibration, forward_limit_mps=.3, steering_limit_rad=.35))
    p = GeometricCoach().propose(d, cfg)
    sweep = p["diagnostics"]["sweeps"]["left"]
    assert sweep["planned_controls"]["drive_mps"] == .3
    assert sweep["planned_controls"]["steering_rad"] == -.35
    assert sweep["latency_s"] == 1.5
    assert sweep["inflation_m"] > .15*1.5
    assert SIMULATOR_GEOMETRY.half_width(.35, 1.) > .09
    assert p["maneuver"] == "stop"


def test_wall_priority_checked_reverse_and_no_reverse_into_blocked_rear():
    cfg = replace(CFG, obstacle_enabled=False)
    stopped = GeometricCoach().propose(diagnostic(position=(3.28, 0.)), cfg)
    assert stopped["maneuver"] == "reverse"
    assert stopped["diagnostics"]["sweeps"]["reverse"]["safe"]
    moving = GeometricCoach().propose(diagnostic(position=(3.28, 0.), speed=.2), cfg)
    assert moving["maneuver"] == "stop"
    blocked = GeometricCoach().propose(diagnostic(position=(.32, 0.), obstacle=(.75, 0.)), CFG)
    assert not blocked["diagnostics"]["sweeps"]["reverse"]["safe"]
    assert blocked["maneuver"] == "stop"


def test_missing_or_invalid_geometry_abstains_instead_of_guessing():
    coach = GeometricCoach()
    assert coach.propose({}, CFG)["text"] is None
    d = diagnostic(speed=math.nan)
    assert coach.propose(d, CFG)["reason"] == "unreliable_geometry"
    d = diagnostic(obstacle=(math.nan, 0.))
    assert coach.propose(d, CFG)["text"] is None
    with pytest.raises(TypeError):
        coach.propose(diagnostic(), None)


def test_geometric_state_json_roundtrip_preserves_maneuver():
    coach = GeometricCoach()
    coach.propose(diagnostic(), CFG)
    saved = json.loads(json.dumps(coach.state_dict(), allow_nan=False))
    restored = GeometricCoach()
    restored.load_state_dict(saved)
    d = diagnostic(now=.5, position=(.6, -.03))
    assert restored.propose(d, CFG) == coach.propose(d, CFG)
    saved["config"]["clearance_margin_m"] = .2
    with pytest.raises(ValueError, match="incompatible"):
        restored.load_state_dict(saved)


def test_one_gate_per_window_no_geometry_when_off_and_no_surviving_cue():
    w, first = enabled()
    draw = w.windows[0]["gate_draw"]
    for i in range(1, 8):
        result = w.advance(i/2, i, diagnostic(now=i/2), CFG)
        assert result["events"] == []
        assert w.windows[0]["gate_draw"] == draw
    assert len(w.windows) == 1
    result = w.advance(4., 8, diagnostic(now=4.), CFG)
    assert len(w.windows) == 2
    assert result["closed_windows"][0]["closed_at_s"] == 4.
    assert w.windows[0]["instructions"][0]["status"] == "expired"
    off = CoachingWindows(1, condition="U", coach=FixedCoach())
    off.coach.propose = lambda *args: pytest.fail("off gate queried privileged geometry")
    assert off.advance(0., 0, {}, CFG)["text"] is None
    assert off.policy_context(20.) == {"instructions": [], "statuses": []}


def test_supersession_not_violation_completion_stays_history_and_single_reward():
    w, first = enabled()
    original = first["instruction_id"]
    w.record_action(receipt(.5), generated_at_s=.3, decision_time_s=.3)
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    result = w.advance(.6, 1, diagnostic(now=.6, speed=.1), CFG)
    replacement = result["instruction_id"]
    assert replacement != original
    assert w.windows[0]["instructions"][0]["status"] == "superseded"
    w.record_action(receipt(1., 2, controls=Control()), generated_at_s=.8)
    w.advance(1.1, 2, diagnostic(now=1.1), CFG)
    assert w.windows[0]["instructions"][-1]["status"] == "completed"
    assert all(x["completed_at"] is not None for x in w.policy_context(2.)["instructions"])
    closed = w.finish(2., transition_id="closing")["closed_windows"][0]
    assert closed["assessable"]
    reward = w.claim_reward(closed["window_id"], "adherent", transition_id="closing")
    assert reward["pay_raw"] == 2. and reward["pay_encoded"] == .2
    replay = w.claim_reward(closed["window_id"], "adherent", transition_id="closing")
    assert replay["pay_raw"] == 0. and not replay["newly_claimed"]
    with pytest.raises(ValueError, match="immutable"):
        w.claim_reward(closed["window_id"], "violated", transition_id="closing")


@pytest.mark.parametrize("generated,decision,received,source,assessable", [
    (.1, .1, .5, "policy", False),  # Generated before delayed availability.
    (.3, .1, .6, "policy", False),  # New inference but old filtered frame.
    (.3, .3, .4, "policy", False),  # Less than .25 s receipt-time reaction.
    (.3, .3, .6, "fallback", False),
    (.3, .3, .6, "policy", True),
])
def test_action_receipt_and_history_cutoff_both_must_follow_availability(generated, decision, received, source, assessable):
    w = CoachingWindows(1, coach=FixedCoach())
    w.update(0., 0, diagnostic(), CFG, available_at_s=.2)
    assert w.policy_context(.1)["instructions"] == []
    assert w.policy_context(.2)["instructions"][0]["issued_at"] == .2
    w.record_action(receipt(received), generated_at_s=generated, decision_time_s=decision, source=source)
    closed = w.finish(1., transition_id="t")["closed_windows"][0]
    assert closed["assessable"] is assessable


def test_geometry_cannot_be_backdated_to_the_triggering_capture():
    w = CoachingWindows(1)
    with pytest.raises(ValueError, match="backdate"):
        w.advance(0., 0, diagnostic(now=.1), CFG)
    assert w.ledger.snapshot()["events"] == []


def test_publication_clock_is_sampled_after_proposal_before_issue():
    calls = []

    class PublicationCoach(FixedCoach):
        def propose(self, diagnostics, config):
            calls.append("proposal")
            return super().propose(diagnostics, config)

    w = CoachingWindows(1, coach=PublicationCoach())

    def clock():
        assert calls == ["proposal"]
        assert not w.ledger.snapshot()["events"]
        calls.append("publication")
        return .5

    result = w.advance(0., 0, diagnostic(), CFG, publication_clock=clock)
    assert calls == ["proposal", "publication"]
    assert result["text"]
    cue = w.windows[0]["instructions"][0]
    assert cue["issued_at_s"] == cue["available_at_s"] == .5
    assert w.ledger.context(.49) == []
    assert w.ledger.context(.5)[0]["issued_at"] == .5
    assert w.windows[0]["scheduled_end_s"] == 4.
    assert w.state_dict()["last_now"] == .5
    json.dumps(w.state_dict(), allow_nan=False)  # No callable enters checkpoints.


@pytest.mark.parametrize("available,expected", [(None, .5), (.2, .5), (.8, .8)])
def test_explicit_availability_never_predates_publication_clock(available, expected):
    w = CoachingWindows(1, coach=FixedCoach())
    w.update(0., 0, diagnostic(), CFG, publication_clock=lambda: .5, available_at_s=available)
    assert w.windows[0]["instructions"][0]["available_at_s"] == expected
    assert w.ledger.context(expected)[0]["issued_at"] == expected


@pytest.mark.parametrize("published", [-1., math.nan, math.inf, .9])
def test_invalid_or_backdated_publication_clock_cannot_issue(published):
    w = CoachingWindows(1, coach=FixedCoach())
    with pytest.raises(ValueError):
        w.advance(1., 0, diagnostic(now=1.), CFG, publication_clock=lambda: published)
    assert w.ledger.snapshot()["events"] == []


def test_publication_clock_not_called_for_off_or_terminal_windows():
    def forbidden():
        pytest.fail("no proposal publication in an off or terminal window")

    off = CoachingWindows(1, condition="U")
    off.advance(0., 0, {}, CFG, publication_clock=forbidden)
    w, _ = enabled()
    w.advance(1., 1, diagnostic(now=1.), CFG, terminal=True, publication_clock=forbidden)


def test_publication_past_deadline_does_not_issue_or_extend_window():
    w = CoachingWindows(1, coach=FixedCoach())
    result = w.advance(0., 0, diagnostic(), CFG, publication_clock=lambda: 4.1)
    assert result["text"] is None and result["events"] == []
    assert len(w.windows) == 1 and w.windows[0]["scheduled_end_s"] == 4.
    assert w.windows[0]["instructions"] == []
    w.advance(4.2, 1, diagnostic(now=4.2), CFG)
    assert len(w.windows) == 2 and w.windows[1]["start_s"] == 4.


@pytest.mark.parametrize("published", [4., 4.010])
def test_publication_at_window_edge_abstains_without_backdating_gate_or_reward(published):
    coach = FixedCoach()
    proposal = deepcopy(coach.proposal)
    coach.proposal["text"] = None
    w = CoachingWindows(1, coach=coach)
    w.advance(0., 0, diagnostic(), CFG)
    coach.proposal = proposal
    original_draw = w.windows[0]["gate_draw"]
    result = w.advance(3.999, 3, diagnostic(now=3.999), CFG, publication_clock=lambda: published)
    assert result["text"] is None and result["events"] == []
    assert len(w.windows) == 1 and w.windows[0]["gate_draw"] == original_draw
    assert w.windows[0]["scheduled_end_s"] == 4.
    assert w.windows[0]["instructions"] == []
    assert w.policy_context(published, scheduled_expiry=True) == {"instructions": [], "statuses": []}
    w.record_action(receipt(published), generated_at_s=3.999, exposed_instruction_ids=[])
    assert len(w.windows) == 2 and w.windows[1]["start_s"] == 4.
    assert w.windows[1]["instructions"] == []  # No retrospective issue on rollover.
    old = w.bind_transition(w.windows[0]["window_id"], "closing")
    assert old["closed_at_s"] == 4. and not old["assessable"]
    assert w.claim_reward(old["window_id"], "adherent", transition_id="closing")["pay_raw"] == 0.


def test_late_safety_publication_cannot_supersede_or_complete_expired_cue():
    w, _ = enabled()
    original_ledger = w.ledger.snapshot()
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    result = w.advance(3.999, 3, diagnostic(now=3.999), CFG, publication_clock=lambda: 4.010)
    assert result["text"] is None and result["events"] == []
    assert w.ledger.snapshot() == original_ledger
    assert w.windows[0]["instructions"][0]["ended_at_s"] is None
    view = w.policy_context(4.010, scheduled_expiry=True)
    assert len(view["instructions"]) == 1 and view["instructions"][0]["completed_at"] == 4.
    assert rendered_policy(view, 4.010)["instructions"][0]["status"] == "completed"


def test_safety_supersession_is_published_at_clock_not_earlier_geometry_time():
    w, _ = enabled()
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    w.advance(.6, 1, diagnostic(now=.6, speed=.1), CFG, publication_clock=lambda: .9)
    assert w.windows[0]["instructions"][0]["ended_at_s"] == .9
    assert len(w.ledger.context(.8)) == 1
    assert w.ledger.context(.8)[0]["completed_at"] is None
    assert len(w.ledger.context(.9)) == 2
    assert w.ledger.context(.9)[0]["completed_at"] == .9


def rendered_policy(context, now):
    import numpy as np
    from numerical_experiment.picar.fix2_data import history_batch
    from numerical_experiment.picar.models import render_context

    batch = history_batch([{"rgb": np.zeros((8, 8, 3), dtype=np.uint8), "time": now,
                            "elapsed": 0., "action": (0., 0., 0., 0.)}],
                          contexts=context["instructions"], statuses=context["statuses"],
                          deadline=20., length=1, width=8, height=8)
    return json.loads(render_context(batch, 0, 0))


def test_scheduled_expiry_view_survives_stall_without_mutating_actual_ledger():
    w, _ = enabled()
    original = w.state_dict()
    legacy = w.policy_context(.5)
    view = w.policy_context(.5, scheduled_expiry=True)
    assert legacy["instructions"][0]["completed_at"] is None
    assert legacy["statuses"] == []
    assert view["instructions"][0]["completed_at"] == 4.
    assert view["statuses"] == [{"text": "Guidance expired: " + view["instructions"][0]["text"],
                                 "available_at": 4., "kind": "robot_status"}]
    before = rendered_policy(view, 3.9)
    after = rendered_policy(view, 4.)
    assert before["instructions"][0]["status"] == "active"
    assert before["status_events"] == []
    assert after["instructions"][0]["status"] == "completed"
    assert after["status_events"][0]["available_at"] == 4.
    assert w.state_dict() == original
    assert w.windows[0]["instructions"][0]["status"] == "active"
    assert w.windows[0]["closed_at_s"] is None
    assert w.policy_context(5., True)["instructions"][0]["completed_at"] == 4.
    view["instructions"][0]["text"] = "caller changed copy"
    assert w.state_dict() == original


def test_scheduled_expiry_uses_real_completion_and_suppresses_obsolete_expiry():
    w, _ = enabled()
    baseline = w.policy_context(1., scheduled_expiry=True)
    w.complete(now=2.)
    original = w.state_dict()
    historic = w.policy_context(1., scheduled_expiry=True)
    assert historic == baseline
    assert historic["instructions"][0]["completed_at"] == 4.
    assert historic["statuses"][0]["available_at"] == 4.
    current = w.policy_context(2., scheduled_expiry=True)
    assert current["instructions"][0]["completed_at"] == 2.
    assert all("expired" not in event["text"] for event in current["statuses"])
    assert w.state_dict() == original


@pytest.mark.parametrize("ending", ["superseded", "terminal", "infrastructure_truncation", "fade_cutoff"])
def test_historical_expiry_view_cannot_reveal_future_actual_ending(ending):
    w, _ = enabled()
    baseline = w.policy_context(1., scheduled_expiry=True)
    if ending == "superseded":
        w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
        w.advance(2., 1, diagnostic(now=2., speed=.1), CFG)
    elif ending == "fade_cutoff":
        w.advance(2., 768, diagnostic(now=2.), CFG)
    else:
        w.finish(2., terminal=ending == "terminal", transition_id="end")
    original = w.state_dict()
    assert w.policy_context(1., scheduled_expiry=True) == baseline
    restored = CoachingWindows(1, coach=FixedCoach())
    restored.load_state_dict(json.loads(json.dumps(original)))
    assert restored.policy_context(1., scheduled_expiry=True) == baseline
    current = w.policy_context(2., scheduled_expiry=True)
    assert current["instructions"][0]["completed_at"] == 2.
    assert not any(e["available_at"] > 2. and baseline["instructions"][0]["text"] in e["text"]
                   for e in current["statuses"])
    assert w.state_dict() == original


def test_scheduled_expiry_only_tracks_current_cue_after_safety_supersession():
    w, _ = enabled()
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    w.advance(1., 1, diagnostic(now=1., speed=.1), CFG)
    view = w.policy_context(1., scheduled_expiry=True)
    assert [c["completed_at"] for c in view["instructions"]] == [1., 4.]
    expiries = [e for e in view["statuses"] if "expired" in e["text"]]
    assert expiries == [{"text": "Guidance expired: Stop driving now.", "available_at": 4.,
                         "kind": "robot_status"}]
    w.advance(2., 768, diagnostic(now=2.), CFG)
    cancelled = w.policy_context(2., scheduled_expiry=True)
    assert cancelled["instructions"][-1]["completed_at"] == 2.
    assert not any(e["available_at"] > 2. for e in cancelled["statuses"])


@pytest.mark.parametrize("exposure,assessable", [(None, True), ([], False), (["not-seen"], False),
                                               (["current"], True)])
def test_explicit_cue_exposure_is_required_in_addition_to_eligible_clocks(exposure, assessable):
    w, first = enabled()
    ids = [first["instruction_id"]] if exposure == ["current"] else exposure
    w.record_action(receipt(.5), generated_at_s=.3, exposed_instruction_ids=ids)
    closed = w.finish(1., transition_id="end")["closed_windows"][0]
    assert closed["assessable"] is assessable
    assert closed["actions"][0]["exposed_instruction_ids"] == ids
    reward = w.claim_reward(closed["window_id"], "adherent", transition_id="end")
    assert reward["pay_raw"] == (2. if assessable else 0.)


def test_exposure_does_not_bypass_publication_or_supersession_timing():
    w = CoachingWindows(1, coach=FixedCoach())
    first = w.advance(0., 0, diagnostic(), CFG, publication_clock=lambda: .5)
    w.record_action(receipt(.8), generated_at_s=.3, decision_time_s=.3,
                    exposed_instruction_ids=[first["instruction_id"]])
    assert not w.finish(1., transition_id="end")["closed_windows"][0]["assessable"]
    w, first = enabled()
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    w.advance(.6, 1, diagnostic(now=.6, speed=.1), CFG)
    w.record_action(receipt(1.), generated_at_s=.8, exposed_instruction_ids=[first["instruction_id"]])
    assert not w.finish(1.5, transition_id="end")["closed_windows"][0]["assessable"]


def test_exposure_list_is_detached_and_survives_checkpoint_and_idempotent_delivery():
    w, first = enabled()
    ids = ["other", first["instruction_id"]]
    r = receipt(.5)
    w.record_action(r, generated_at_s=.3, exposed_instruction_ids=ids)
    ids.clear()
    restored = CoachingWindows(1, coach=FixedCoach())
    restored.load_state_dict(json.loads(json.dumps(w.state_dict())))
    assert not restored.record_action(r, generated_at_s=.3,
                                       exposed_instruction_ids=[first["instruction_id"], "other"])
    assert restored.finish(1., transition_id="end")["closed_windows"][0]["assessable"]
    with pytest.raises(ValueError, match="conflicting"):
        restored.record_action(r, generated_at_s=.3, exposed_instruction_ids=[])


@pytest.mark.parametrize("ids", ["cue", {}, [None], [""]])
def test_invalid_explicit_exposure_is_rejected_without_recording_action(ids):
    w, _ = enabled()
    with pytest.raises(ValueError, match="exposed instruction IDs"):
        w.record_action(receipt(.5), generated_at_s=.3, exposed_instruction_ids=ids)
    assert w.windows[0]["actions"] == []


def test_delayed_receipt_owns_historical_window_without_retrospective_cues():
    w, _ = enabled()
    # A command receipt can be harvested after a scheduled boundary.
    w.record_action(receipt(4.2), generated_at_s=3.8)
    assert len(w.windows) == 2 and w.windows[1]["instructions"] == []
    assert w.windows[0]["closed_at_s"] == 4.
    assert not w.windows[0]["assessable"]
    # A still-later delivery of an earlier receipt fills the closed owner.
    w.record_action(receipt(3.5, 2), generated_at_s=3.2)
    assert w.windows[0]["assessable"]
    assert [a["accepted_at_s"] for a in w.windows[0]["actions"]] == [3.5]
    w.advance(4.5, 4, diagnostic(now=4.5), CFG, transition_id="physical-4")
    reward = w.claim_reward(w.windows[0]["window_id"], "violated", transition_id="physical-4")
    assert reward["pay_encoded"] == -.2
    with pytest.raises(ValueError, match="immutable"):
        w.record_action(receipt(3.8, 3), generated_at_s=3.3)


def test_gate_replay_json_checkpoint_does_not_redraw_or_duplicate_receipts():
    w, _ = enabled()
    r = receipt(.5)
    w.record_action(r, generated_at_s=.3)
    saved = json.loads(json.dumps(w.state_dict(), allow_nan=False))
    copy = CoachingWindows(1, coach=FixedCoach())
    copy.load_state_dict(saved)
    assert not copy.record_action(r, generated_at_s=.3)
    for now in (1., 4., 4.5, 8., 13.):
        a = w.advance(now, int(now), diagnostic(now=now), CFG)
        b = copy.advance(now, int(now), diagnostic(now=now), CFG)
        assert a == b
    assert copy.state_dict() == w.state_dict()
    assert len(w.windows) == 4
    incompatible = CoachingWindows(1, coach=FixedCoach(), window_s=6.)
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.load_state_dict(saved)


@pytest.mark.parametrize("complete_first,assessable", [(False, False), (True, True)])
def test_fade_cutoff_cancels_unfinished_but_preserves_completed_response(complete_first, assessable):
    w = CoachingWindows(1, coach=FixedCoach())
    w.advance(0., 0., diagnostic(), CFG)
    w.record_action(receipt(.5), generated_at_s=.3)
    if complete_first:
        w.complete(now=.7)
    result = w.advance(1., 768, diagnostic(now=1.), CFG, transition_id="cutoff")
    closed = result["closed_windows"][0]
    assert closed["close_reason"] == "fade_cutoff"
    assert closed["assessable"] is assessable
    assert result["text"] is None
    assert w.windows[-1]["probability"] == 0.
    assert w.policy_context(1.)["instructions"][0]["completed_at"] is not None
    for now in (2., 6., 11.):
        assert w.advance(now, 900, diagnostic(now=now), CFG)["text"] is None


@pytest.mark.parametrize("terminal,assessable", [(True, True), (False, False)])
def test_true_terminal_can_be_scored_infrastructure_truncation_cancels(terminal, assessable):
    w, _ = enabled()
    w.record_action(receipt(.5), generated_at_s=.3)
    closed = w.finish(.6, terminal, transition_id="end")["closed_windows"][0]
    assert closed["assessable"] is assessable
    assert not w.finish(.6, terminal, transition_id="end")["closed_windows"]


def test_exact_boundary_cutoff_draws_only_one_gate_for_the_new_window():
    w, _ = enabled()
    result = w.advance(4., 768, diagnostic(now=4.), CFG, transition_id="cutoff")
    assert len(result["closed_windows"]) == 1
    assert result["closed_windows"][0]["close_reason"] == "fade_cutoff"
    assert len(w.windows) == 2
    assert w.windows[1]["probability"] == 0.


def test_receipt_rolling_exact_cutoff_cannot_preserve_unfinished_bonus_or_redraw():
    w, _ = enabled()
    w.record_action(receipt(.5), generated_at_s=.3)
    w.record_action(receipt(4., 2), generated_at_s=3.8)
    assert w.windows[0]["assessable"]
    next_draw = w.windows[1]["gate_draw"]
    result = w.advance(4., 768, diagnostic(now=4.), CFG, transition_id="cutoff")
    assert len(w.windows) == 2
    assert w.windows[1]["gate_draw"] == next_draw
    assert result["text"] is None
    assert result["closed_windows"][0]["close_reason"] == "fade_cutoff"
    assert not result["closed_windows"][0]["assessable"]


def test_superseded_but_uncompleted_response_is_cancelled_at_fade_cutoff():
    w, _ = enabled()
    w.record_action(receipt(.5), generated_at_s=.3)
    w.coach.proposal.update(text="Stop driving now.", maneuver="stop", safety=True)
    w.advance(.7, 1, diagnostic(now=.7, speed=.1), CFG)
    result = w.advance(1., 768, diagnostic(now=1.), CFG, transition_id="cutoff")
    assert not result["closed_windows"][0]["assessable"]


def test_no_response_terminal_is_unassessable_and_completed_reward_survives_resume():
    no_response, _ = enabled()
    assert not no_response.finish(.1, transition_id="early")["closed_windows"][0]["assessable"]
    w, _ = enabled()
    w.record_action(receipt(.5), generated_at_s=.3)
    window = w.finish(.8, transition_id="end")["closed_windows"][0]
    w.claim_reward(window["window_id"], "adherent", transition_id="end")
    restored = CoachingWindows(1, coach=FixedCoach())
    restored.load_state_dict(json.loads(json.dumps(w.state_dict())))
    replay = restored.claim_reward(window["window_id"], "adherent", transition_id="end")
    assert replay["pay_raw"] == replay["pay_encoded"] == 0.


def test_receipt_at_terminal_cutoff_has_no_remaining_response_opportunity():
    w, _ = enabled()
    w.finish(.8, transition_id="end")
    w.record_action(receipt(.8), generated_at_s=.3)
    assert not w.compliance_window(w.windows[0]["window_id"])["assessable"]


def test_auto_complete_alignment_and_reverse_use_measured_motion_not_reward():
    cfg = replace(CFG, obstacle_enabled=False)
    w = CoachingWindows(1)
    first = w.advance(0., 0, diagnostic(yaw=-.3), cfg)
    assert "Steer left" in first["text"]
    w.record_action(receipt(.5), generated_at_s=.3)
    w.advance(.7, 1, diagnostic(now=.7), cfg)
    assert w.windows[0]["instructions"][0]["status"] == "completed"
    reverse = CoachingWindows(1)
    assert reverse.advance(0., 0, diagnostic(position=(3.28, 0.)), cfg)["text"].startswith("Reverse")
    reverse.record_action(receipt(.5, controls=Control(drive_mps=-.1)), generated_at_s=.3)
    reverse.advance(1.4, 1, diagnostic(now=1.4, position=(3.18, 0.)), cfg)
    assert reverse.windows[0]["instructions"][0]["status"] == "completed"


def test_policy_and_compliance_language_whitelists_exclude_geometry_and_audit():
    w, _ = enabled()
    w.record_action(receipt(.5), generated_at_s=.3)
    closed = w.finish(1., transition_id="end")["closed_windows"][0]
    context = w.policy_context(1.)
    evidence = w.compliance_window(closed["window_id"])
    forbidden = {"reason", "diagnostics", "position_m", "route_side", "planned_controls", "audits", "layout", "condition"}

    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for v in value.values():
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)

    check(context)
    check(evidence)
    assert context["instructions"][0]["text"] == closed["instructions"][0]["text"]
    assert context != {"instructions": [], "statuses": []}


@pytest.fixture
def bullet_sim():
    pytest.importorskip("pybullet")
    sim = PyBulletBackend(CFG, run_id="fix6-unit-audit")
    sim.reset(seed=4)
    try:
        yield sim
    finally:
        sim.close()


def place(sim, position, yaw=0., *, obstacle=None):
    """Administrative unit fixture, never demonstration or policy execution."""
    p, cid = sim._p, sim.client_id
    p.resetBasePositionAndOrientation(sim.vehicle_id, (*position, .045),
                                     p.getQuaternionFromEuler((0, 0, yaw)), physicsClientId=cid)
    p.resetBaseVelocity(sim.vehicle_id, (0, 0, 0), (0, 0, 0), physicsClientId=cid)
    if obstacle is not None:
        sim._layout["obstacle"] = obstacle
        p.resetBasePositionAndOrientation(sim.obstacle_id, (*obstacle, .045), (0, 0, 0, 1), physicsClientId=cid)
    p.performCollisionDetection(physicsClientId=cid)


def execute(sim, controls, seconds):
    observation = sim.observe()
    command = RobotCommand(sim.run_id, sim.session_id, 0, observation.observation_id,
                           controls, sim.time_s, min(seconds, sim.config.control_interval_s),
                           sim.time_s+sim.config.control_interval_s)
    received = sim.command(command)
    assert received.accepted
    assert received.executed_controls == controls
    result = sim.advance(round(seconds/sim.config.physics_dt_s), training=False)
    return received, result


def execute_held(sim, controls, seconds):
    """Repeated real protocol receipts keep control live, never bypass motors."""
    receipts = []
    for sequence in range(math.ceil(seconds/sim.config.control_interval_s)):
        observation = sim.observe()
        duration = min(sim.config.control_interval_s, seconds-sequence*sim.config.control_interval_s)
        command = RobotCommand(sim.run_id, sim.session_id, sequence, observation.observation_id,
                               controls, sim.time_s, duration, sim.time_s+duration)
        received = sim.command(command)
        assert received.accepted and received.executed_controls == controls
        receipts.append(received)
        result = sim.advance(round(duration/sim.config.physics_dt_s), training=False)
        if result.terminated or result.truncated:
            break
    return receipts, sim.diagnostics()


def test_declared_geometry_matches_real_bullet_collision_shapes(bullet_sim):
    sim, g = bullet_sim, SIMULATOR_GEOMETRY
    aabbs = [sim._p.getAABB(w, physicsClientId=sim.client_id) for w in sim.wall_ids]
    assert g.arena_bounds_m == pytest.approx((aabbs[0][1][0], aabbs[1][0][0], aabbs[2][1][1], aabbs[3][0][1]))
    lower, upper = sim._p.getAABB(sim.obstacle_id, physicsClientId=sim.client_id)
    assert g.barrier_half_extents_m == pytest.approx(((upper[0]-lower[0])/2, (upper[1]-lower[1])/2))
    shape = sim._p.getCollisionShapeData(sim.vehicle_id, -1, physicsClientId=sim.client_id)[0]
    assert shape[3][:2] == pytest.approx((2*g.chassis_half_length_m, 2*g.chassis_half_width_m))


@pytest.mark.parametrize("steer,drive,expected_yaw", [(-1, 1, 1), (1, 1, -1), (-1, -1, -1), (1, -1, 1)])
def test_forward_reverse_sign_audit_uses_executed_bullet_motion(bullet_sim, steer, drive, expected_yaw):
    sim = bullet_sim
    place(sim, (.6, 0.))
    before = sim.diagnostics()
    controls = Control(steering_rad=steer*CFG.calibration.steering_limit_rad,
                       drive_mps=CFG.calibration.forward_limit_mps if drive > 0 else -CFG.calibration.reverse_limit_mps)
    received, _ = execute(sim, controls, .5)
    after = sim.diagnostics()
    measured_yaw = math.atan2(math.sin(chassis_yaw(after)-chassis_yaw(before)),
                              math.cos(chassis_yaw(after)-chassis_yaw(before)))
    # Ground truth is independent measured physics and actual receipt controls.
    assert expected_yaw*measured_yaw > .015
    assert drive*(after["position_m"][0]-before["position_m"][0]) > .02
    predicted = bicycle_step((.6, 0., 0.), received.executed_controls, .5, steering_gain=after["steering_gain"])
    assert expected_yaw*predicted[2] > 0
    assert after["unsafe_contact"] is False


def test_center_clear_but_chassis_corner_contacts_barrier_in_real_simulator(bullet_sim):
    sim = bullet_sim
    ox, oy = sim.diagnostics()["layout"]["obstacle"]
    place(sim, (ox, oy+.30))
    d = sim.diagnostics()
    assert d["position_m"][1] > oy+SIMULATOR_GEOMETRY.barrier_half_extents_m[1]
    assert d["unsafe_contact"]  # Center-point clearance is not vehicle clearance.
    p = GeometricCoach().propose(d, CFG)
    assert p["maneuver"] == "stop"
    assert p["diagnostics"]["current_clearance"]["barrier_m"] < 0


def test_checked_wall_reverse_executes_without_contact_in_real_simulator(bullet_sim):
    sim = bullet_sim
    place(sim, (3.28, 0.))
    before = sim.diagnostics()
    proposal = GeometricCoach().propose(before, CFG)
    assert proposal["maneuver"] == "reverse"
    controls = Control(**proposal["diagnostics"]["sweeps"]["reverse"]["planned_controls"])
    received, _ = execute(sim, controls, .5)
    after = sim.diagnostics()
    assert received.executed_controls.drive_mps < 0
    assert after["position_m"][0] < before["position_m"][0]-.02
    assert not after["unsafe_contact"]
    assert sim.exposure_intervals == 0.  # Diagnostics never acquire training examples.


@pytest.mark.parametrize("side", [-1, 1])
def test_passage_turn_cue_has_correct_measured_yaw_in_real_simulator(bullet_sim, side):
    sim = bullet_sim
    place(sim, (.5, 0.), obstacle=(1.575, -side*.2))
    before = sim.diagnostics()
    proposal = GeometricCoach().propose(before, CFG)
    assert proposal["diagnostics"]["route_side"] == side
    turn = "left" if side == 1 else "right"
    control = Control(**proposal["diagnostics"]["sweeps"][turn]["planned_controls"])
    received, _ = execute(sim, control, .5)
    after = sim.diagnostics()
    measured = chassis_yaw(after)-chassis_yaw(before)
    # The audit label comes from executed dynamics, not the proposal's predicate.
    assert side*measured > .02
    assert received.executed_controls.steering_rad*side < 0
    assert not after["unsafe_contact"]


@pytest.mark.parametrize("case", ["wall", "barrier_swept_turn"])
def test_projected_unsafe_sweeps_are_audited_against_actual_collision(bullet_sim, case):
    sim = bullet_sim
    if case == "wall":
        place(sim, (3.28, 0.))
        key = "forward"
    else:
        place(sim, (1.05, .35), obstacle=(1.575, 0.))
        key = "right"
    before = sim.diagnostics()
    assert not before["unsafe_contact"]
    proposal = GeometricCoach().propose(before, CFG)
    sweep = proposal["diagnostics"]["sweeps"][key]
    assert not sweep["safe"]
    receipts, after = execute_held(sim, Control(**sweep["planned_controls"]), 1.5)
    assert all(r.accepted for r in receipts)
    assert after["unsafe_contact"] and after["end_reason"] == "collision"
    assert sim.exposure_intervals == 0.

"""All-choice FIFO complement during a real Blimp turn after Egg->Romantic.

Original Copy and Scoocher occupy12..15; the real Egg/R occupies12/14.
Copy sees the raw Egg and gets no Romantic power (H02). Its earlier real
Scoocher qualifications still execute after it changes away from Scoocher.
The explicit S-before-C inverse closure naturally exhausts at36 controls
and77 drainable configurations. Only its complement is certified, after
all actual old jobs run, full mechanics return, and every Star/choice binds.
This module does not execute game actions or mutate the queue or ledger.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
import hashlib
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .scoring import ScoreLedger, ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS


from .static_inheritance import static_inheritance_evidence

REASON = "egg_copy_scoocher_all_choice_fifo_cycle_proved"


def _powers(copy, romantic, scoocher):
    highest = max(copy, romantic, scoocher)
    result = []
    if romantic == highest:
        result.append(False)
    if scoocher == highest:
        result.append(True)
    return tuple(result or (False,))


def _successors(control, job, profile="SC"):
    """Complete single-job algebra with the actual S-before-C FIFO profile."""
    copy, romantic, scoocher, power = control
    children = []
    if profile not in {"SC", "CS"}:
        raise ValueError("unknown Scoocher/Copy FIFO profile")
    for label in profile:
        if label == "S" and job != "S" or label == "C" and job != "C" and power:
            children.append(label)
    if job == "C":
        copy = 12 + (copy - 12 + 1) % 4
        arrived = copy
    elif job == "S":
        scoocher = 12 + (scoocher - 12 + 1) % 4
        arrived = scoocher
    else:
        romantic = 12 + (romantic - 12 + 2) % 4
        arrived = romantic
    # This is EXACTLY two, including R's own arrival; three is not two.
    if sum(position == arrived for position in (copy, romantic, scoocher)) == 2:
        children.append("R")
    return tuple(((copy, romantic, scoocher, after), "".join(children))
                 for after in _powers(copy, romantic, scoocher))


@lru_cache(maxsize=2)
def _finite_proof(profile="SC"):
    controls = tuple((c, r, s, p) for c, r, s in product(range(12, 16), (12, 14), range(12, 16))
                     for p in _powers(c, r, s))
    incoming = defaultdict(list)
    for control in controls:
        for job in "CSR":
            for target, output in _successors(control, job, profile):
                if target not in controls:
                    raise ValueError("preparing FIFO algebra escaped its complete control domain")
                incoming[target].append((control, job, output))
    bad = {(control, "") for control in controls}
    todo = deque(sorted(bad))
    # No search limit or queue-length cutoff. This finite least fixed point
    # is constructed to natural exhaustion, then every inverse edge is checked.
    while todo:
        control, word = todo.popleft()
        for previous, job, output in incoming[control]:
            if word.endswith(output):
                tail = word[:-len(output)] if output else word
                candidate = previous, job + tail
                if candidate not in bad:
                    bad.add(candidate)
                    todo.append(candidate)
    for control, word in bad:
        for previous, job, output in incoming[control]:
            if word.endswith(output):
                tail = word[:-len(output)] if output else word
                if (previous, job + tail) not in bad:
                    raise ValueError("preparing FIFO inverse fixed point is not closed")
    table = [[list(control), word] for control, word in sorted(bad)]
    digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (len(controls) != 36 or len(bad) != 77 or max(len(word) for _, word in bad) != 4
            or digest != "1459833012a4901b96d0dce6d7f585eee2bad1f0b498b854229762d05f140c85"):
        raise ValueError("preparing FIFO exact algebra differs from its independently verified certificate")
    return frozenset(controls), frozenset(bad), digest


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _actors(engine):
    actors = tuple(engine.racers.values())
    if (len(actors) != 4 or any(not isinstance(actor.ability, Ability) for actor in actors)
            or {actor.ability for actor in actors} != {Ability.BLIMP, Ability.COPY_CAT, Ability.EGG, Ability.SCOOCHER}):
        return None
    roles = {actor.ability: actor for actor in actors}
    result = tuple(roles[power] for power in (Ability.BLIMP, Ability.COPY_CAT, Ability.EGG, Ability.SCOOCHER))
    z, c, r, s = result
    return result


def _profile(engine, actors):
    copy, scoocher = actors[1], actors[3]
    return "".join("C" if rid == copy.racer_id else "S" for rid in engine._clockwise_order()
                   if rid in {copy.racer_id, scoocher.racer_id})


def _initial_evidence(engine, actors):
    _, _, romantic, _ = actors
    # Raw Egg remains visible to Copy. A borrowed/effective-R flag cannot
    # substitute for the globally authenticated physical acquisition.
    if romantic.inherited_ability != Ability.ROMANTIC:
        return None
    evidence = static_inheritance_evidence(engine, romantic, Ability.ROMANTIC)
    if not evidence:
        return None
    binding = json.loads(evidence[0])
    grants = binding["inherit_events"]
    if (len(grants) != 1 or grants[0]["racer_id"] != romantic.racer_id
            or grants[0]["ability"] != Ability.ROMANTIC.value):
        return None
    return grants[0]["sequence"], evidence


def _queue_valid(engine, pending, actors, inherited_sequence, *, require_unconsumed=True):
    z, copy, romantic, scoocher = actors
    c, r, s = (actor.racer_id for actor in (copy, romantic, scoocher))
    if any(not isinstance(job, PendingEffect) or type(job.queued_sequence) is not int
           or type(job.trigger_sequence) is not int or type(job.steps) is not int
           or type(job.racer_id) is not int or type(job.source_id) is not int
           or job.target_id is not None and type(job.target_id) is not int for job in pending):
        return False
    ids = [job.queued_sequence for job in pending]
    if not ids or ids != sorted(set(ids)):
        return False
    needed = {job.trigger_sequence for job in pending}
    positions = {item["racer_id"]: item["position"] for item in engine.initial_racers}
    copy_power, at_trigger = None, {}
    batch_sizes = defaultdict(int)
    for event in engine.events:
        if event.kind in {"move", "warp"}:
            positions[event.racer_id] = event.after
        elif event.kind == "copy_changed" and event.racer_id == c:
            copy_power = event.ability
        elif event.kind == "stop":
            batch_sizes[event.arrival_batch_id] += 1
        if event.sequence in needed:
            at_trigger[event.sequence] = dict(positions), copy_power
    if (positions != {rid: actor.position for rid, actor in engine.racers.items()}
            or copy_power != copy.copied_ability):
        return False
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}}
    for job in pending:
        if require_unconsumed and job.queued_sequence in consumed:
            return False
        if (type(job.queued_sequence) is not int or type(job.trigger_sequence) is not int
                or type(job.steps) is not int
                or not inherited_sequence < job.trigger_sequence < job.queued_sequence <= len(engine.events)):
            return False
        cause, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if job.racer_id == r:
            if ((job.source_id, job.kind, job.steps, job.ability, job.target_id) != (r, "move", 2, Ability.ROMANTIC.value, r)
                    or cause.kind != "stop" or cause.racer_id not in {c, r, s}
                    or type(cause.after) is not int or not 0 <= cause.after < engine.track.finish_position
                    or type(cause.before) is not int or type(cause.parent_sequence) is not int
                    or not 1 <= cause.parent_sequence < cause.sequence
                    or type(cause.arrival_batch_id) is not int or cause.arrival_batch_id < 1
                    or not _event(cause, "stop", cause.racer_id, before=cause.before, after=cause.after,
                                  parent_sequence=cause.parent_sequence, arrival_batch_id=cause.arrival_batch_id)
                    or batch_sizes[cause.arrival_batch_id] != 1
                    or sum(position == cause.after for position in at_trigger[cause.sequence][0].values()) != 2
                    or not _event(queued, "effect_queued", r, source_id=r, target_id=r,
                                  ability=Ability.ROMANTIC.value, parent_sequence=cause.sequence)):
                return False
        else:
            if (job.racer_id not in {c, s} or job.source_id not in ({r, s} if job.racer_id == c else {c, r})
                    or (job.kind, job.steps, job.ability, job.target_id) != ("move", 1, Ability.SCOOCHER.value, None)):
                return False
            power = Ability.ROMANTIC.value if job.source_id == r else Ability.SCOOCHER.value
            if (not _event(cause, "ability_effect", job.source_id, source_id=job.source_id, ability=power,
                           amount=2 if job.source_id == r else 1, parent_sequence=cause.parent_sequence)
                    or type(cause.parent_sequence) is not int or not 1 <= cause.parent_sequence < cause.sequence
                    or engine.events[cause.parent_sequence - 1].kind != "effect_queued"
                    or engine.events[cause.parent_sequence - 1].racer_id != job.source_id
                    or job.racer_id == c and at_trigger[cause.sequence][1] != Ability.SCOOCHER.value
                    or not _event(queued, "effect_queued", job.racer_id, source_id=job.source_id,
                                  ability=Ability.SCOOCHER.value, parent_sequence=cause.sequence)):
                return False
    return True


def key(engine):
    if (engine.track != WILD_WILDS or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies or hasattr(engine, "_arrow_visits")
            or hasattr(engine, "_arrow_queue_visits") or getattr(engine, "_movement_depth", 0) != 0
            or getattr(engine, "_duel_offered_epochs", {})
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
            or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
            or type(getattr(engine, "_resolution_depth", 0)) is not int or engine._resolution_depth < 1):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    z, copy, romantic, scoocher = actors
    if (z.position != 4 or romantic.position not in (12, 14)
            or copy.position not in range(12, 16) or scoocher.position not in range(12, 16)):
        return None
    highest = max(copy.position, romantic.position, scoocher.position)
    allowed = {actor.racer_id: None if actor is romantic else Ability.SCOOCHER
               for actor in (romantic, scoocher) if actor.position == highest}
    if not allowed:
        allowed = {None: None}
    if copy.copied_target not in allowed or copy.copied_ability != allowed[copy.copied_target]:
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE or type(actor.racer_id) is not int or type(actor.position) is not int
                or type(actor.tripped) is not bool or actor.pre_race_resolved is not True
                or type(actor.turns_taken) is not int or actor.turns_taken < 0
                or actor.finish_place is not None or actor.finish_places or actor.prediction is not None
                or actor is not romantic and actor.inherited_ability is not None
                or actor is not copy and (actor.copied_target is not None or actor.copied_ability is not None)):
            return None
    try:
        evidence = _initial_evidence(engine, actors)
        if evidence is None or not _queue_valid(engine, tuple(engine.pending_effects), actors, evidence[0]):
            return None
        labels = {copy.racer_id: "C", romantic.racer_id: "R", scoocher.racer_id: "S"}
        control = copy.position, romantic.position, scoocher.position, copy.copied_ability == Ability.SCOOCHER
        profile = _profile(engine, actors)
        if (control, "".join(labels[job.racer_id] for job in engine.pending_effects)) in _finite_proof(profile)[1]:
            return None
        roster = tuple((rid, tuple((item.name, getattr(actor, item.name)) for item in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                              "deck": engine.initial_remaining_abilities}, sort_keys=True)
        state = (engine.track, engine._state_key(include_scores=False, include_pending=False, include_decisions=False),
                 roster, engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._preparing, engine._turn_in_progress,
                 engine.turn_index, tuple(engine.turns), engine._resolution_depth,
                 getattr(engine, "_physical_resolution_depth", 0), initial, engine._score_start,
                 engine.initial_code_fingerprint, engine.max_events, profile, evidence)
        hash(state)
        return state
    except (KeyError, ValueError, TypeError, IndexError, AttributeError):
        return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple[PendingEffect, ...]
    score_before: ScoreSnapshot
    arrival_counter: int
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        raise ValueError("not a certified normal Egg/Copy/Scoocher FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine.score_ledger.snapshot(), getattr(engine, "_arrival_counter", 0),
                    tuple(engine.events), tuple(engine.decisions))


class _Reject(Exception):
    pass


class _Reader:
    """Authenticate real events against the proved algebra, without actions."""

    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot, self.actors = engine, snapshot, actors
        self.z, self.c, self.r, self.s = (actor.racer_id for actor in actors)
        self.order = tuple(engine._clockwise_order())
        self.profile = _profile(engine, actors)
        self.positions = {actor.racer_id: actor.position for actor in actors}
        self.target = actors[1].copied_target
        self.power = actors[1].copied_ability
        self.events = engine.events[snapshot.event_start:]
        self.offset, self.decision = 0, snapshot.decision_start
        self.batch = snapshot.arrival_counter
        self.balances = dict(snapshot.score_before.totals)
        self.children, self.score_events, self.choice_events = [], [], []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def queued(self, actor, source, cause, romantic=False):
        if romantic:
            event = self.take("effect_queued", actor, source_id=source, target_id=actor,
                              ability=Ability.ROMANTIC.value, parent_sequence=cause.sequence)
            pending = PendingEffect(actor, source, cause.sequence, event.sequence,
                                    "move", 2, Ability.ROMANTIC.value, actor)
        else:
            event = self.take("effect_queued", actor, source_id=source,
                              ability=Ability.SCOOCHER.value, parent_sequence=cause.sequence)
            pending = PendingEffect(actor, source, cause.sequence, event.sequence)
        self.children.append(pending)

    def arrival(self, actor, destination, steps, cause, *, arrow=False):
        before = self.positions[actor]
        passed = [rid for rid in self.order if rid != actor
                  and (before < self.positions[rid] < destination or before > self.positions[rid] > destination)]
        self.batch += 1
        power = Ability.ROMANTIC.value if actor == self.r else Ability.SCOOCHER.value
        values = {"before": before, "after": destination, "amount": steps,
                  "movement_kind": "track" if arrow else "ability", "parent_sequence": cause.sequence,
                  "arrival_batch_id": self.batch}
        if not arrow:
            values.update(source_id=actor, ability=power)
        move = self.take("move", actor, **values)
        self.positions[actor] = destination
        for other in passed:
            self.take("passing", actor, target_id=other, parent_sequence=move.sequence, arrival_batch_id=self.batch)
        stop = self.take("stop", actor, before=before, after=destination,
                         parent_sequence=move.sequence, arrival_batch_id=self.batch)
        if sum(position == destination for position in self.positions.values()) == 2:
            self.queued(self.r, self.r, stop, romantic=True)
        if destination == 13:
            balance = self.balances[actor]
            score = self.take("score", actor, before=balance, after=balance + 1, amount=1,
                              parent_sequence=stop.sequence, reason="star")
            self.balances[actor] += 1
            self.score_events.append(score)
        if destination == 16:
            self.arrival(actor, 12, -4, stop, arrow=True)

    def refresh(self):
        highest = max(self.positions.values())
        targets = tuple(rid for rid in self.order if rid != self.c and self.positions[rid] == highest)
        old = self.target, self.power
        if self.target in targets:
            legal = ((Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets))
                     if len(targets) > 1 else ())
        elif len(targets) > 1:
            legal = tuple(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets)
        else:
            legal = ()
            self.target = targets[0] if targets else None
        if legal:
            if self.decision >= len(self.engine.decisions):
                raise _Reject
            record = self.engine.decisions[self.decision]
            public = []
            by_id = {actor.racer_id: actor for actor in self.actors}
            for rid in self.engine.turn_order:
                actor = by_id[rid]
                ability = self.power if rid == self.c else Ability.ROMANTIC if rid == self.r else actor.ability
                public.append(PublicRacer(rid, actor.name, self.positions[rid], "active",
                                         ability.value if ability else None, actor.tripped, None,
                                         self.target if rid == self.c else None))
            observation = Observation(
                self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
                WILD_WILDS.name, WILD_WILDS.finish_position, WILD_WILDS.second_corner_position,
                tuple(sorted(WILD_WILDS.trip_positions)), tuple(public), (), None, None,
                score_totals=tuple((rid, self.balances[rid]) for rid, _ in self.snapshot.score_before.totals))
            request = DecisionRequest(self.decision + 1, self.c, DecisionWindow.COPY_CHOICE, observation, legal)
            if record.request != request or record.action not in legal:
                raise _Reject
            choice = self.take("decision", self.c, source_id=record.action.source_id, target_id=record.action.target_id,
                               action_kind=record.action.kind.value, decision_id=request.decision_id)
            if record.event_sequence != choice.sequence:
                raise _Reject
            self.choice_events.append(choice.sequence)
            self.decision += 1
            if record.action.kind != ActionKind.KEEP:
                self.target = record.action.target_id
        self.power = Ability.SCOOCHER if self.target == self.s else None
        if old != (self.target, self.power):
            self.take("copy_changed", self.c, source_id=self.c, target_id=self.target,
                      ability=self.power.value if self.power else None)

    def control(self):
        return self.positions[self.c], self.positions[self.r], self.positions[self.s], self.power == Ability.SCOOCHER

    def job(self, pending):
        before = self.control()
        actor = pending.racer_id
        label = {self.c: "C", self.r: "R", self.s: "S"}[actor]
        self.children = []
        romantic = actor == self.r
        power = self.take("ability_effect", actor, source_id=actor,
                          ability=Ability.ROMANTIC.value if romantic else Ability.SCOOCHER.value,
                          amount=2 if romantic else 1, parent_sequence=pending.queued_sequence)
        # Notifications precede movement and use the OLD Copy power.
        for rid in self.order:
            if rid == self.s and actor != self.s or rid == self.c and actor != self.c and self.power == Ability.SCOOCHER:
                self.queued(rid, actor, power)
        steps = 2 if romantic else 1
        self.arrival(actor, self.positions[actor] + steps, steps, power)
        self.refresh()
        word = "".join({self.c: "C", self.r: "R", self.s: "S"}[job.racer_id] for job in self.children)
        if (self.control(), word) not in _successors(before, label, self.profile):
            raise _Reject
        return tuple(self.children)


def completed(engine, snapshot):
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    evidence = snapshot.state[-1]
    if not _queue_valid(engine, snapshot.original_pending, actors, evidence[0], require_unconsumed=False):
        return False
    current = engine.score_ledger.snapshot()
    old = snapshot.score_before
    if current.player_ids != old.player_ids or current.entries[:len(old.entries)] != old.entries:
        return False
    events = engine.events[snapshot.event_start:]
    if (not events or any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(events))):
        return False
    reader, queue = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
    labels = {actors[1].racer_id: "C", actors[2].racer_id: "R", actors[3].racer_id: "S"}
    controls, bad, proof_sha = _finite_proof(_profile(engine, actors))
    executed, counts = [], {"C": 0, "R": 0, "S": 0}
    try:
        while reader.offset < len(events):
            word = "".join(labels[job.racer_id] for job in queue)
            if not queue or reader.control() not in controls or (reader.control(), word) in bad:
                return False
            pending = queue.popleft()
            children = reader.job(pending)
            executed.append(pending.queued_sequence)
            counts[labels[pending.racer_id]] += 1
            queue.extend(children)
    except (KeyError, ValueError, _Reject):
        return False
    original_ids = {job.queued_sequence for job in snapshot.original_pending}
    remaining_ids = [job.queued_sequence for job in queue]
    final_word = "".join(labels[job.racer_id] for job in queue)
    if (reader.decision != len(engine.decisions) or reader.batch != getattr(engine, "_arrival_counter", 0)
            or not original_ids.issubset(executed) or original_ids.intersection(remaining_ids)
            or any(index <= snapshot.event_start for index in remaining_ids)
            or tuple(queue) != tuple(engine.pending_effects) or not queue
            or (reader.control(), final_word) in bad
            or reader.positions != {actor.racer_id: actor.position for actor in actors}
            or (reader.target, reader.power) != (actors[1].copied_target, actors[1].copied_ability)
            or reader.balances != engine.score_ledger.totals()
            or counts["C"] % 4 or counts["S"] % 4 or counts["R"] % 2
            or counts["C"] + counts["S"] < 4):
        return False
    added = current.entries[len(old.entries):]
    if (not added or len(added) != len(reader.score_events)
            or any((entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source, entry.reason,
                     entry.event_sequence, entry.phase) != (
                        len(old.entries) + index + 1, event.racer_id, engine.race_index, 1, "track", "star",
                        event.sequence, "during_race")
                   for index, (entry, event) in enumerate(zip(added, reader.score_events)))):
        return False
    races = {race: dict(totals) for race, totals in old.race_totals}
    for entry in added:
        races[entry.race_index][entry.player_id] += entry.delta
    expected_races = tuple((race, tuple((rid, races[race][rid]) for rid, _ in totals)) for race, totals in old.race_totals)
    if current.race_totals != expected_races:
        return False
    return {"original_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
            "executed_pending_ids": executed, "new_pending_ids": remaining_ids,
            "queue_before": len(snapshot.original_pending), "queue_after": len(queue),
            "job_counts": counts, "copy_decision_event_ids": reader.choice_events,
            "star_point_increments": {str(rid): reader.balances[rid] - total for rid, total in old.totals},
            "abstract_control_count": len(controls), "drainable_configurations": len(bad),
            "maximum_drainable_word_length": 4, "drainable_table_sha256": proof_sha,
            "inherit_event_sequence": evidence[0], "fifo_profile": _profile(engine, actors),
            "static_romantic": {"racer_id": actors[2].racer_id, "original_ability": Ability.EGG.value,
                                "inherited_ability": Ability.ROMANTIC.value, "acquisition": evidence[1]},
            "mechanical_state_returned": True, "all_original_jobs_executed": True,
            "all_actual_copy_choices_retained": True, "only_fresh_pending_descendants_stopped": True,
            "proof": "exact all-choice FIFO backward fixed-point complement; full actual cohort and mechanical/Star ledger authenticated"}

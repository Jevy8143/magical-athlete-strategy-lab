"""Narrow native Party/Scoocher/Baby/Duelist Wild22 FIFO authentication.

S always replaces itself by S_B,D. KEEP removes only a D; a real Duel
adds S_D,D for either winner of all36 dice pairs. This nonempty S invariant
does not assume that a policy eventually chooses KEEP. The conservative
runtime certificate additionally waits for an actual KEEP epoch return, so
suspended original Duel continuations unwind without an invented choice or
an epoch assignment. Party's main die and movement are still owed.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
from functools import wraps
import hashlib
import json

from .decisions import Action, ActionKind, DecisionRecord, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import WILD_WILDS
from .romantic_baby_shared12_cycle import _ENGINE_FIELDS, _EVENT_FIELDS, _RACER_FIELDS, _event, _known_pending, _ledger_valid
from .fish_baby_arrow11_cycle import _typed_request, _Reject as _BadRequest

REASON = "native_party_scoocher_baby_duel22_fifo_cycle_proved"
ATTRIBUTE = "_scoocher_baby_duel22_scope"
_ROLES = (Ability.PARTY_ANIMAL, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.DUELIST)
_ACTUAL = frozenset({"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                     "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"})


class _Reject(Exception):
    pass


def _need(value):
    if not value:
        raise _Reject


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _past_digest(events, decisions):
    """Value binding also detects in-place mutation of nominally frozen data."""
    value = {"events": [asdict(event) for event in events], "decisions": [asdict(record) for record in decisions]}
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _actors(engine):
    _need(type(engine.racers) is dict and len(engine.racers) == 4)
    roles = {r.ability: r for r in engine.racers.values()}
    _need(set(roles) == set(_ROLES))
    return tuple(roles[role] for role in _ROLES)


def _epoch(engine):
    return (len(engine.turns), next((e.sequence for e in reversed(engine.events) if e.kind in _ACTUAL), 0))


def _base(engine, actors, *, current_actor=None, flexible_baby=False):
    _need(not (set(vars(engine)) - (_ENGINE_FIELDS | {ATTRIBUTE})))
    _need(engine.track == WILD_WILDS and engine.termination_reason is None and engine.finish_order == []
          and type(engine.finish_order) is list and type(engine.pending_effects) is deque
          and engine._preparing is False and engine._turn_in_progress is True
          and type(engine._copy_locks) is set and not engine._copy_locks and engine._refreshing_copies is False
          and getattr(engine, "_inert_duel_romantic", None) is None and engine.next_turn_racer_id is None
          and engine._genius_prediction is None and type(engine.score_ledger) is ScoreLedger)
    _need(type(engine.policies) is dict and type(engine.policy_metadata) is dict)
    _need(all(type(getattr(engine, name)) is list for name in
              ("turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn")))
    _need(type(engine.initial_remaining_abilities) is tuple and type(engine.previous_winners) is tuple)
    _need(type(engine.turn_index) is int and 0 <= engine.turn_index < 4
          and len(engine.turn_order) == 4 and set(engine.turn_order) == set(engine.racers)
          and all(type(rid) is int for rid in engine.turn_order)
          and engine.turn_order[engine.turn_index] == (actors[0].racer_id if current_actor is None else current_actor))
    _need(engine.seed is None or type(engine.seed) is int)
    _need(type(engine.race_index) is int and 1 <= engine.race_index <= 4
          and type(engine.max_events) is int and engine.max_events > 0
          and type(engine._score_start) is int and engine._score_start >= 0)
    _need(not hasattr(engine, "max_action_events") or type(engine.max_action_events) is int and engine.max_action_events > 0)
    _need(all(type(getattr(engine, name, None)) is int and getattr(engine, name) > 0
              for name in ("_resolution_depth", "_physical_resolution_depth"))
          and type(getattr(engine, "_arrival_counter", 0)) is int
          and getattr(engine, "_arrival_counter", 0) >= (0 if flexible_baby else 1))
    _need(type(getattr(engine, "_movement_depth", 0)) is int and getattr(engine, "_movement_depth", 0) == 0)
    offered = getattr(engine, "_duel_offered_epochs", {})
    _need(type(offered) is dict and set(offered) <= {actors[3].racer_id})
    _need(all(type(key) is int and type(value) is tuple and len(value) == 2
              and all(type(n) is int and n >= 0 for n in value) for key, value in offered.items()))
    for actor in actors:
        _need(type(actor) is Racer and set(vars(actor)) == _RACER_FIELDS
              and type(actor.racer_id) is int and engine.racers.get(actor.racer_id) is actor
              and type(actor.ability) is Ability and type(actor.status) is RacerStatus
              and actor.status == RacerStatus.ACTIVE and actor.pre_race_resolved is True
              and type(actor.position) is int and type(actor.tripped) is bool
              and type(actor.turns_taken) is int and actor.turns_taken >= 0
              and actor.inherited_ability is None and actor.copied_target is None and actor.copied_ability is None
              and actor.prediction is None and actor.finish_place is None
              and type(actor.finish_places) is tuple and not actor.finish_places)
    _need(tuple(actor.tripped for actor in actors) == (False, False, actors[2].tripped if flexible_baby else True, False))
    _need(all(type(event) is RaceEvent and type(event.sequence) is int and event.sequence == i + 1 and set(vars(event)) == _EVENT_FIELDS
              for i, event in enumerate(engine.events)))
    ids = [event.decision_id for event in engine.events if event.kind == "decision"]
    _need(ids == list(range(1, len(engine.decisions) + 1))
          and all(type(record) is DecisionRecord for record in engine.decisions))
    _need(all(_known_pending(job) for job in engine.pending_effects))


def _mechanics(engine, actors, *, party_position=None, actor_position=None, handoff=False):
    roster = [{**asdict(r), "status": r.status.value} for _, r in sorted(engine.racers.items())]
    if party_position is not None:
        next(row for row in roster if row["racer_id"] == actors[0].racer_id)["position"] = party_position
    if actor_position is not None:
        rid, position = actor_position
        next(row for row in roster if row["racer_id"] == rid)["position"] = position
    return _json({"roster": roster,
        "scores": asdict(engine.score_ledger.snapshot()), "turns": [asdict(t) for t in engine.turns],
        "order": engine.turn_order, "index": engine.turn_index, "race": engine.race_index,
        "remaining": engine.remaining_abilities, "initial_remaining": engine.initial_remaining_abilities,
        "egg": engine.egg_drawn, "winners": engine.previous_winners,
        "initial_racers": engine.initial_racers, "initial_scores": engine.initial_score_ledger,
        "metadata": engine.policy_metadata, "policy_seats": sorted(engine.policies),
        "score_start": engine._score_start, "fingerprint": engine.initial_code_fingerprint,
        "seed": engine.seed, "budget": engine.max_events, "action_budget": getattr(engine, "max_action_events", None),
        "physical_depth": engine._physical_resolution_depth,
        "known_fields": sorted(name for name in (set(vars(engine)) | {"_arrival_counter", "_movement_depth", "_duel_offered_epochs"})
                              if not (handoff and name == "_inert_duel_romantic"))})


@dataclass(slots=True)
class Frame:
    trigger: int | None
    start: int
    depth: int
    physical: int
    yielded: list[int] = field(default_factory=list)
    source_id: int | None = None


@dataclass(frozen=True, slots=True)
class Snapshot:
    event_start: int
    decision_start: int
    original_pending: tuple
    prefix_events: tuple
    prefix_decisions: tuple
    mechanics: str
    arrival: int
    raw_epoch: tuple
    owner_ids: tuple
    owner_depth: int


@dataclass(frozen=True, slots=True)
class Handoff:
    """Created only after the real first Scope's original generators return."""
    certificate: RaceEvent
    events: tuple
    decisions: tuple
    mechanics: str
    arrival: int
    raw_epoch: tuple
    owner_returns: tuple
    completion_checked: bool
    prefix_sha256: str
    current_actor: int | None = None
    start_position: int = 21
    origin: str = "certified_Party_pull"


@dataclass(slots=True)
class Scope:
    root_start: int
    root_end: int
    root_arrival: int
    root_pending: tuple
    root_events: tuple
    root_decisions: tuple
    drains: list = field(default_factory=list)
    duels: list = field(default_factory=list)
    returned: list = field(default_factory=list)
    snapshot: Snapshot | None = None
    disabled: bool = False
    closed: bool = False
    certificate_event: int | None = None
    expected_returns: list = field(default_factory=list)
    actual_returns: list = field(default_factory=list)
    phase: str = "before_main"
    handoff: Handoff | None = None
    current_actor: int | None = None
    root_mechanics: str | None = None
    root_prefix_sha256: str | None = None


class _Reader:
    def __init__(self, engine, actors, start, batch, positions, *, after_main=False):
        self.engine, self.actors, self.offset, self.batch = engine, actors, start, batch
        self.p, self.s, self.b, self.d = (r.racer_id for r in actors)
        self.positions = dict(positions)
        self.executed, self.counts, self.duel_yields, self.keeps = [], Counter(), {}, []
        self.duel_starts = {}
        self.targets = (tuple(rid for rid in engine._clockwise_order() if rid in {self.p, self.s})
                        if after_main else (self.s,))

    def take(self, kind, actor, **values):
        _need(self.offset < len(self.engine.events))
        event = self.engine.events[self.offset]
        _need(_event(event, kind, actor, **values) and event.sequence == self.offset + 1)
        self.offset += 1
        if kind in {"move", "stop"}:
            self.positions[actor] = event.after
        return event

    def move(self, actor, before, after, amount, parent, *, ability=None, source=None, kind="ability"):
        _need(self.positions[actor] == before)
        self.batch += 1
        moved = self.take("move", actor, before=before, after=after, amount=amount, parent_sequence=parent,
                          movement_kind=kind, ability=ability, source_id=source, arrival_batch_id=self.batch)
        return self.take("stop", actor, before=before, after=after, parent_sequence=moved.sequence, arrival_batch_id=self.batch)

    def queued_s(self, source, effect):
        event = self.take("effect_queued", self.s, source_id=source, ability="Scoocher", parent_sequence=effect.sequence)
        return PendingEffect(self.s, source, effect.sequence, event.sequence)

    def queued_d(self, stop):
        event = self.take("effect_queued", self.d, source_id=self.d, target_id=self.d,
                          ability="Duelist", parent_sequence=stop.sequence)
        return PendingEffect(self.d, self.d, stop.sequence, event.sequence, "duel", 0, "Duelist", self.d)

    def baby_arrow(self, actor, stop):
        self.batch += 1
        self.take("move", actor, before=23, after=25, amount=2, movement_kind="track",
                  parent_sequence=stop.sequence, arrival_batch_id=self.batch)
        baby = self.take("ability_effect", self.b, source_id=self.b, ability="Huge Baby",
                         parent_sequence=stop.sequence, arrival_batch_id=self.batch)
        child = self.queued_s(self.b, baby)
        self.batch += 1
        back = self.take("stop", actor, before=25, after=24, parent_sequence=baby.sequence, arrival_batch_id=self.batch)
        return child, self.move(actor, 24, 22, -2, back.sequence, kind="track")

    def choice(self, parent, targets):
        event = self.engine.events[self.offset]
        _need(type(event.decision_id) is int and 1 <= event.decision_id <= len(self.engine.decisions))
        record = self.engine.decisions[event.decision_id - 1]
        try:
            _typed_request(record)
        except _BadRequest:
            raise _Reject
        actors = {actor.racer_id: actor for actor in self.actors}
        public = tuple(PublicRacer(rid, actors[rid].name, self.positions[rid], "active", actors[rid].ability.value,
                                  actors[rid].tripped, None, None) for rid in self.engine.turn_order)
        obs = Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1, self.engine.track.name, 30, 15,
                          tuple(sorted(self.engine.track.trip_positions)), public, (), None, None,
                          score_totals=tuple(self.engine.score_ledger.totals().items()))
        targets = targets if type(targets) is tuple else (targets,)
        legal = (Action(ActionKind.KEEP), *(Action(ActionKind.DUEL, self.d, target) for target in targets))
        _need(record.request == DecisionRequest(event.decision_id, self.d, DecisionWindow.DUEL, obs, legal)
              and record.action in legal and record.event_sequence == event.sequence)
        action = record.action
        self.take("decision", self.d, source_id=action.source_id, target_id=action.target_id,
                  action_kind=action.kind.value, decision_id=event.decision_id, parent_sequence=parent)
        if action.kind == ActionKind.KEEP:
            self.keeps.append((event.sequence, next((e.sequence for e in reversed(self.engine.events[:event.sequence - 1]) if e.kind in _ACTUAL), 0)))
        return action

    def duel(self, pending, target, *, root=False):
        start = self.offset
        identity = None if pending is None else pending.queued_sequence
        self.duel_starts[identity] = start
        action = self.choice(identity, target)
        if action.kind == ActionKind.KEEP:
            _need(not root)
            self.counts["D_KEEP"] += 1
            return ()
        target = action.target_id
        effect = self.take("ability_effect", self.d, source_id=self.d, target_id=target, ability="Duelist",
                           parent_sequence=self.engine.events[start].sequence)
        child = self.queued_s(self.d, effect)
        dice = []
        for actor in (self.d, target):
            event = self.engine.events[self.offset]
            _need(type(event.amount) is int and 1 <= event.amount <= 6)
            dice.append(event.amount)
            self.take("duel_roll", actor, source_id=self.d, ability="Duelist", amount=event.amount, parent_sequence=effect.sequence)
        winner = self.d if dice[0] >= dice[1] else target
        if root:
            _need(winner == self.d and self.positions[winner] == 21)
            stop = self.move(winner, 21, 23, 2, effect.sequence, ability="Duelist", source=self.d)
            baby, _ = self.baby_arrow(winner, stop)
            result = (child, baby)
        else:
            stop = self.move(winner, 22, 24, 2, effect.sequence, ability="Duelist", source=self.d)
            returned = self.move(winner, 24, 22, -2, stop.sequence, kind="track")
            result = (child, self.queued_d(returned))
        self.duel_yields[identity] = self.offset
        self.counts["D_DUEL_winner_" + str(winner)] += 1
        return result

    def job(self, pending):
        _need(_known_pending(pending))
        self.executed.append(pending.queued_sequence)
        if pending.racer_id == self.s:
            _need(pending.source_id in {self.b, self.d} and pending.kind == "move" and pending.steps == 1
                  and pending.ability == "Scoocher" and pending.target_id is None)
            effect = self.take("ability_effect", self.s, source_id=self.s, ability="Scoocher", amount=1,
                               parent_sequence=pending.queued_sequence)
            stop = self.move(self.s, 22, 23, 1, effect.sequence, ability="Scoocher", source=self.s)
            child, returned = self.baby_arrow(self.s, stop)
            self.counts["S"] += 1
            return (child, self.queued_d(returned))
        _need((pending.racer_id, pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id)
              == (self.d, self.d, "duel", 0, "Duelist", self.d))
        return self.duel(pending, self.targets)


def _root_pending(engine, actors, start, end):
    """Re-derive every original qualification from the same real Party root."""
    p, s, b, d = (r.racer_id for r in actors)
    _need(type(start) is int and type(end) is int and start >= 0 and end == start + 9 and end <= len(engine.events))
    positions = {row["racer_id"]: row["position"] for row in engine.initial_racers}
    _need(len(positions) == 4 and set(positions) == set(engine.racers))
    for event in engine.events[:start]:
        if event.kind in {"move", "warp"}:
            _need(event.before == positions[event.racer_id])
            positions[event.racer_id] = event.after
        elif event.kind == "stop":
            _need(positions[event.racer_id] in {event.before, event.after})
            positions[event.racer_id] = event.after
        elif event.kind in {"inherit", "copy_changed", "finish", "eliminate"}:
            raise _Reject
    _need(positions == {p: 21, s: 18, b: 26, d: 22} and _ledger_valid(engine, start))
    batch = engine.events[start + 2].arrival_batch_id
    _need(type(batch) is int and batch > 0)
    reader = _Reader(engine, actors, start, batch - 1, positions)
    power = reader.take("ability_effect", p, source_id=p, ability="Party Animal")
    first = reader.queued_s(p, power)
    reader.batch += 1
    moves = []
    targets = {s: (18, 19), b: (26, 25), d: (22, 21)}
    for actor in (rid for rid in engine._clockwise_order() if rid != p):
        before, after = targets[actor]
        moved = reader.take("move", actor, before=before, after=after, amount=after - before, source_id=p,
                            ability="Party Animal", movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=reader.batch)
        moves.append(moved)
    stops = [reader.take("stop", e.racer_id, before=e.before, after=e.after, parent_sequence=e.sequence,
                         arrival_batch_id=reader.batch) for e in moves]
    second = reader.queued_d(next(stop for stop in stops if stop.racer_id == d))
    _need(reader.offset == end)
    return (first, second), reader.batch


def _birth(engine, actors):
    _base(engine, actors)
    _need(tuple(r.position for r in actors) == (21, 19, 25, 21) and len(engine.events) >= 9)
    start, end = len(engine.events) - 9, len(engine.events)
    pending, arrival = _root_pending(engine, actors, start, end)
    _need(tuple(engine.pending_effects) == pending and engine._arrival_counter == arrival)
    return Scope(start, end, arrival, pending, tuple(engine.events), tuple(engine.decisions))


def _birth_direct(engine, actors, source, trigger):
    """Native depth-one None-trigger opportunity comes from _start_turn_impl.

    A queued Duel always has its real integer parent and a parent drain.
    This distinct owner is never represented as a synthetic PendingEffect.
    """
    current = engine.turn_order[engine.turn_index]
    _base(engine, actors, current_actor=current, flexible_baby=True)
    _need(trigger is None and source is actors[3] and not engine.pending_effects
          and engine._resolution_depth == engine._physical_resolution_depth == 1
          and tuple(actor.position for actor in actors) == (22, 22, 25, 22)
          and current in {actor.racer_id for actor in actors[1:]}
          and _ledger_valid(engine, len(engine.events)))
    # Match the native increment of this turn; no completed turn is invented.
    initial = {row["racer_id"]: row["turns_taken"] for row in engine.initial_racers}
    for actor in actors:
        _need(actor.turns_taken == initial[actor.racer_id]
              + sum(turn.racer_id == actor.racer_id for turn in engine.turns)
              + (actor.racer_id == current))
    return Scope(len(engine.events), len(engine.events), getattr(engine, "_arrival_counter", 0), (),
        tuple(engine.events), tuple(engine.decisions), phase="direct_turn_start", current_actor=current,
        root_mechanics=_mechanics(engine, actors, handoff=True),
        root_prefix_sha256=_past_digest(engine.events, engine.decisions))


def _make_handoff(engine, scope):
    actors = _actors(engine)
    direct = scope.phase == "direct_turn_start"
    _base(engine, actors, current_actor=scope.current_actor, flexible_baby=direct)
    _need(scope.phase in {"before_main", "direct_turn_start"} and not scope.drains and not scope.duels
          and not scope.expected_returns and not engine.pending_effects
          and tuple(actor.position for actor in actors) == ((22, 22, 25, 22) if direct else (21, 22, 25, 22)))
    event = engine.events[-1]
    if scope.closed:
        packet = json.loads(event.operation)
        _need(event.sequence == scope.certificate_event and event.reason == REASON
              and scope.actual_returns == packet["Duel_continuations_still_to_return"])
    else:
        # A real direct KEEP is a finite exit, never a cycle certificate. It
        # can still authenticate the genuinely subsequent own-main movement.
        _need(direct and scope.returned == [None] and len(engine.events) == scope.root_end + 1
              and _past_digest(engine.events[:scope.root_end], engine.decisions[:len(scope.root_decisions)]) == scope.root_prefix_sha256
              and _mechanics(engine, actors, handoff=True) == scope.root_mechanics)
        reader = _Reader(engine, actors, scope.root_end, scope.root_arrival,
                         {actor.racer_id: actor.position for actor in actors}, after_main=True)
        _need(reader.duel(None, reader.targets) == () and reader.offset == len(engine.events))
    _need(engine._duel_offered_epochs.get(actors[3].racer_id) == _epoch(engine))
    return Handoff(certificate=event, events=tuple(engine.events), decisions=tuple(engine.decisions),
        mechanics=_mechanics(engine, actors, handoff=True), arrival=getattr(engine, "_arrival_counter", 0),
        raw_epoch=tuple(sorted(engine._duel_offered_epochs.items())),
        owner_returns=tuple((identity, event.sequence) for identity in scope.actual_returns),
        completion_checked=True, prefix_sha256=_past_digest(engine.events, engine.decisions),
        current_actor=scope.current_actor,
        start_position=engine.racers[scope.current_actor].position if direct else 21,
        origin=("certified_direct_Duel" if scope.closed else "finite_direct_KEEP") if direct else "certified_Party_pull")


def _main_pending(engine, actors, handoff, end):
    """Authenticate real prior returns and the one actual Party main root."""
    _need(type(handoff) is Handoff and handoff.completion_checked is True
          and type(handoff.arrival) is int and type(handoff.events) is tuple and type(handoff.decisions) is tuple
          and (handoff.current_actor is None or type(handoff.current_actor) is int)
          and type(handoff.start_position) is int and type(handoff.origin) is str
          and type(handoff.owner_returns) is tuple
          and all(type(item) is tuple and len(item) == 2
                  and (item[0] is None or type(item[0]) is int) and type(item[1]) is int
                  for item in handoff.owner_returns)
          and tuple(engine.events[:len(handoff.events)]) == handoff.events
          and tuple(engine.decisions[:len(handoff.decisions)]) == handoff.decisions
          and _past_digest(engine.events[:len(handoff.events)], engine.decisions[:len(handoff.decisions)]) == handoff.prefix_sha256
          and handoff.events and handoff.events[-1] == handoff.certificate)
    p, s, b, d = (actor.racer_id for actor in actors)
    direct = handoff.origin in {"certified_direct_Duel", "finite_direct_KEEP"}
    _need(direct or handoff.origin == "certified_Party_pull")
    current = p if handoff.current_actor is None else handoff.current_actor
    _need(current == engine.turn_order[engine.turn_index]
          and ((not direct and handoff.current_actor is None and handoff.start_position == 21)
               or (direct and current in {s, d} and handoff.start_position == 22)))
    _need(tuple(actor.position for actor in actors) == (22, 22, 25, 22))
    event = handoff.certificate
    _need(type(event) is RaceEvent and event.sequence == len(handoff.events))
    if handoff.origin == "finite_direct_KEEP":
        _need(_event(event, "decision", d, action_kind="keep", decision_id=len(handoff.decisions))
              and not handoff.owner_returns and handoff.raw_epoch == ((d, (len(engine.turns), next(
                  (item.sequence for item in reversed(handoff.events[:-1]) if item.kind in _ACTUAL), 0))),))
    else:
        packet = json.loads(event.operation)
        _need(_event(event, "loop_stopped", current, reason=REASON, parent_sequence=packet["capture_event_count"] + 1,
                         amount=len(packet["new_pending_ids"]), operation=_json(packet))
              and (packet["current_main_die_and_move_still_owed"] is True if direct else packet["Party_main_die_and_move_still_owed"] is True))
        _need(packet["epoch_mutated_by_certificate"] is False
              and handoff.owner_returns == tuple((identity, event.sequence) for identity in packet["Duel_continuations_still_to_return"])
              and handoff.raw_epoch == tuple(sorted((int(rid), tuple(value)) for rid, value in packet["raw_epoch_at_return"].items())))
    _need(_mechanics(engine, actors, actor_position=(current, handoff.start_position), handoff=True) == handoff.mechanics)
    start = event.sequence
    _need(type(end) is int and start < end <= len(engine.events))
    positions = {p: 22, s: 22, b: 25, d: 22}
    positions[current] = handoff.start_position
    reader = _Reader(engine, actors, start, handoff.arrival, positions, after_main=True)
    amount = engine.events[start].amount
    _need(type(amount) is int and amount in ((1, 2, 3) if direct else (1, 2, 3, 4)))
    roll = reader.take("roll", current, amount=amount)
    locked = reader.take("roll_locked", current, amount=amount, parent_sequence=roll.sequence)
    resolved = reader.take("main_move_resolved", current, before=amount, after=amount, parent_sequence=locked.sequence)
    reader.batch += 1
    origin, destination = handoff.start_position, handoff.start_position + amount
    moved = reader.take("move", current, before=origin, after=destination, amount=amount, movement_kind="main",
                        parent_sequence=resolved.sequence, arrival_batch_id=reader.batch)
    for target in (rid for rid in engine._clockwise_order() if not direct and amount > 1 and rid in {s, d}):
        reader.take("passing", current, target_id=target, parent_sequence=moved.sequence, arrival_batch_id=reader.batch)
    child = None
    if destination == 25:
        baby = reader.take("ability_effect", b, source_id=b, ability="Huge Baby", parent_sequence=resolved.sequence,
                           arrival_batch_id=reader.batch)
        child = reader.queued_s(b, baby)
        reader.batch += 1
        stop = reader.take("stop", current, before=25, after=24, parent_sequence=baby.sequence, arrival_batch_id=reader.batch)
        returned = reader.move(current, 24, 22, -2, stop.sequence, kind="track")
    else:
        stop = reader.take("stop", current, before=origin, after=destination, parent_sequence=moved.sequence, arrival_batch_id=reader.batch)
        if destination == 23:
            child, returned = reader.baby_arrow(current, stop)
        elif destination == 24:
            returned = reader.move(current, 24, 22, -2, stop.sequence, kind="track")
        else:
            returned = stop
    duel = reader.queued_d(returned)
    _need(reader.offset == end and reader.positions == {p: 22, s: 22, b: 25, d: 22})
    return ((child, duel) if child is not None else (duel,)), reader.batch


def _birth_after_main(engine, actors, handoff):
    _need(type(handoff) is Handoff)
    _base(engine, actors, current_actor=handoff.current_actor, flexible_baby=handoff.current_actor is not None)
    pending, arrival = _main_pending(engine, actors, handoff, len(engine.events))
    _need(tuple(engine.pending_effects) == pending and engine._arrival_counter == arrival
          and tuple(sorted(engine._duel_offered_epochs.items())) == handoff.raw_epoch)
    return Scope(handoff.certificate.sequence, len(engine.events), arrival, pending,
                 tuple(engine.events), tuple(engine.decisions), phase="after_main", handoff=handoff,
                 current_actor=handoff.current_actor)


def _authenticate(engine, scope):
    actors = _actors(engine)
    _need(type(scope) is Scope)
    direct = scope.phase == "direct_turn_start"
    _base(engine, actors, current_actor=scope.current_actor, flexible_baby=scope.current_actor is not None)
    _need(type(scope.closed) is bool and type(scope.disabled) is bool
          and type(scope.root_start) is int and type(scope.root_end) is int
          and (scope.current_actor is None or type(scope.current_actor) is int)
          and tuple(engine.events[:scope.root_end]) == scope.root_events
          and tuple(engine.decisions[:len(scope.root_decisions)]) == scope.root_decisions
          and type(scope.root_pending) is tuple and (len(scope.root_pending) == 0 if direct else 1 <= len(scope.root_pending) <= 2)
          and all(_known_pending(job) for job in scope.root_pending) and _ledger_valid(engine, scope.root_start))
    _need(scope.phase in {"before_main", "after_main", "direct_turn_start"})
    if direct:
        _need(type(scope.current_actor) is int and scope.current_actor in {actor.racer_id for actor in actors[1:]}
              and scope.root_start == scope.root_end and type(scope.root_arrival) is int
              and scope.root_arrival >= 0
              and _past_digest(engine.events[:scope.root_end], engine.decisions[:len(scope.root_decisions)]) == scope.root_prefix_sha256
              and _mechanics(engine, actors, handoff=True) == scope.root_mechanics)
    else:
        root_pending, root_arrival = (_root_pending(engine, actors, scope.root_start, scope.root_end)
            if scope.phase == "before_main" else _main_pending(engine, actors, scope.handoff, scope.root_end))
        _need(scope.root_pending == root_pending and type(scope.root_arrival) is int and scope.root_arrival == root_arrival)
    p, s, b, d = (r.racer_id for r in actors)
    if scope.phase == "before_main":
        reader = _Reader(engine, actors, scope.root_end, scope.root_arrival, {p: 21, s: 19, b: 25, d: 21})
        first, second = scope.root_pending
        power = reader.take("ability_effect", s, source_id=s, ability="Scoocher", amount=1, parent_sequence=first.queued_sequence)
        reader.move(s, 19, 20, 1, power.sequence, ability="Scoocher", source=s)
        old_s, baby_s = reader.duel(second, p, root=True)
        power = reader.take("ability_effect", s, source_id=s, ability="Scoocher", amount=1, parent_sequence=old_s.queued_sequence)
        reader.move(s, 20, 21, 1, power.sequence, ability="Scoocher", source=s)
        power = reader.take("ability_effect", s, source_id=s, ability="Scoocher", amount=1, parent_sequence=baby_s.queued_sequence)
        stop = reader.move(s, 21, 22, 1, power.sequence, ability="Scoocher", source=s)
        duel = reader.queued_d(stop)
        fifo = deque(reader.duel(duel, s))
    else:
        reader = _Reader(engine, actors, scope.root_end, scope.root_arrival,
                         {p: 22, s: 22, b: 25, d: 22}, after_main=True)
        fifo = deque(reader.duel(None, reader.targets) if direct else scope.root_pending)
    _need(fifo and (scope.phase in {"after_main", "direct_turn_start"} or any(job.racer_id == s for job in fifo)))
    cuts = {reader.offset: tuple(fifo)}
    while reader.offset < len(engine.events):
        _need(fifo)
        fifo.extend(reader.job(fifo.popleft()))
        cuts[reader.offset] = tuple(fifo)
    _need(reader.offset == len(engine.events) and tuple(fifo) == tuple(engine.pending_effects)
          and reader.batch == engine._arrival_counter
          and reader.positions == {r.racer_id: r.position for r in actors})
    _need(scope.drains and len(scope.drains) == len(scope.duels) + (0 if direct else 1)
          and scope.drains[-1].depth == engine._resolution_depth
          and all(type(frame) is Frame and type(frame.depth) is int and type(frame.physical) is int
                  and type(frame.start) is int and frame.start >= 0 and type(frame.yielded) is list
                  and all(type(value) is int for value in frame.yielded)
                  and frame.physical == engine._physical_resolution_depth for frame in (*scope.drains, *scope.duels)))
    for i, owner in enumerate(scope.duels):
        child = scope.drains[i if direct else i + 1]
        _need((owner.trigger is None if direct and i == 0 else type(owner.trigger) is int)
              and owner.source_id == d and type(owner.source_id) is int
              and owner.start == reader.duel_starts.get(owner.trigger)
              and (owner.depth == 1 if direct and i == 0 else owner.depth == scope.drains[i - 1 if direct else i].depth + 1)
              and child.depth == owner.depth + 1
              and owner.yielded == [reader.duel_yields.get(owner.trigger)]
              and child.trigger == owner.trigger and child.start == owner.yielded[0]
              and child.source_id is None)
    return actors, reader, cuts


def capture(engine, scope):
    try:
        actors, reader, _ = _authenticate(engine, scope)
        d, s = actors[3].racer_id, actors[1].racer_id
        offered = getattr(engine, "_duel_offered_epochs", {})
        _need(reader.keeps and reader.keeps[-1][0] == len(engine.events) and offered.get(d) == _epoch(engine)
              and any(job.racer_id == s for job in engine.pending_effects))
        return Snapshot(len(engine.events), len(engine.decisions), tuple(engine.pending_effects), tuple(engine.events),
                        tuple(engine.decisions), _mechanics(engine, actors), engine._arrival_counter,
                        tuple(sorted(offered.items())), tuple(owner.trigger for owner in scope.duels), engine._resolution_depth)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
        return None


def completed(engine, snapshot):
    try:
        scope = getattr(engine, ATTRIBUTE)
        _need(type(snapshot) is Snapshot and type(scope) is Scope and not scope.closed
              and len(engine.events) > snapshot.event_start)
        actors, reader, cuts = _authenticate(engine, scope)
        d = actors[3].racer_id
        capture_epoch = (len(engine.turns), next(e.sequence for e in reversed(snapshot.prefix_events[:-1]) if e.kind in _ACTUAL))
        _need(type(snapshot.event_start) is int and type(snapshot.decision_start) is int
              and type(snapshot.arrival) is int and type(snapshot.owner_depth) is int
              and type(snapshot.original_pending) is tuple and all(_known_pending(job) for job in snapshot.original_pending)
              and snapshot.raw_epoch == ((d, capture_epoch),))
        _need(tuple(engine.events[:snapshot.event_start]) == snapshot.prefix_events
              and tuple(engine.decisions[:snapshot.decision_start]) == snapshot.prefix_decisions
              and _mechanics(engine, actors) == snapshot.mechanics
              and cuts[snapshot.event_start] == snapshot.original_pending)
        last = engine.events[-1]
        _need(last.kind == "decision" and last.action_kind == "keep" and last.racer_id == d
              and reader.keeps[-1][0] == len(engine.events) and getattr(engine, "_duel_offered_epochs", {}).get(d) == _epoch(engine))
        suffix_reader = _Reader(engine, actors, snapshot.event_start, snapshot.arrival,
                                {actor.racer_id: actor.position for actor in actors}, after_main=scope.phase != "before_main")
        fifo, served = deque(snapshot.original_pending), []
        while suffix_reader.offset < len(engine.events):
            _need(fifo)
            job = fifo.popleft()
            served.append(job.queued_sequence)
            fifo.extend(suffix_reader.job(job))
        originals = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        _need([identity for identity in served if identity in originals] == originals
              and not set(originals).intersection(fresh) and tuple(fifo) == tuple(engine.pending_effects)
              and suffix_reader.batch == engine._arrival_counter)
        owners = tuple(owner.trigger for owner in scope.duels)
        extra = owners[len(snapshot.owner_ids):]
        _need(owners[:len(snapshot.owner_ids)] == snapshot.owner_ids
              and engine._resolution_depth == snapshot.owner_depth + 2 * len(extra)
              and all(identity in served and identity in suffix_reader.duel_yields for identity in extra))
        original_owners = set(snapshot.owner_ids) | set(originals)
        returning = list(reversed(owners))
        # A real KEEP is the last actual choice; the certificate is not an
        # actual epoch event, so all these saved while guards return naturally.
        result = {"capture_event_count": snapshot.event_start, "original_pending_ids": originals,
                "executed_pending_ids": served, "new_pending_ids": fresh,
                "original_Duel_bodies_really_executed": True,
                "Duel_continuations_still_to_return": returning,
                "original_Duel_continuations_still_to_return": [identity for identity in returning if identity in original_owners],
                "fresh_Duel_continuations_still_to_return": [identity for identity in returning if identity not in original_owners],
                "suspended_Duel_body_evidence": [{"pending_id": identity,
                    "origin": "direct_turn_start" if identity is None else "ancestor" if identity in snapshot.owner_ids else "captured_original" if identity in originals else "fresh_descendant",
                    "first_decision_event": reader.duel_starts[identity] + 1,
                    "complete_real_body_end": reader.duel_yields[identity]} for identity in owners],
                "raw_epoch_at_capture": dict(snapshot.raw_epoch), "raw_epoch_at_return": dict(engine._duel_offered_epochs),
                "epoch_equality_at_both_cuts": True, "epoch_mutated_by_certificate": False,
                "owner_resolution_depth": snapshot.owner_depth, "actual_resolution_depth": engine._resolution_depth,
                "physical_resolution_depth": engine._physical_resolution_depth,
                "arrival_counter_at_capture": snapshot.arrival, "arrival_counter_at_return": engine._arrival_counter,
                "typed_job_counts": dict(suffix_reader.counts), "all_choice_invariant": "S->S_B,D; D KEEP->empty; D DUEL all36 dice->S_D,D; queued S never decreases",
                "Party_main_die_and_move_still_owed": scope.phase == "before_main", "mechanical_roster_and_full_ledger_returned": True}
        if scope.current_actor is not None:
            result.update(current_actor_id=scope.current_actor,
                current_main_die_and_move_still_owed=scope.phase == "direct_turn_start" and not engine.racers[scope.current_actor].tripped,
                current_recovery_still_owed=scope.phase == "direct_turn_start" and engine.racers[scope.current_actor].tripped,
                current_turn_completion_still_owed=True,
                phase="before_real_current_main" if scope.phase == "direct_turn_start" else "after_real_current_main",
                all_choice_invariant="S->S_B,D; D KEEP->empty; DUEL Party or S, all72 target/dice branches->S_D,D; queued S never decreases")
        if scope.phase == "after_main":
            result.update(phase="after_real_Party_main" if scope.current_actor is None else "after_real_current_main", Party_turn_completion_still_owed=scope.current_actor is None,
                authentic_first_Scope_completion_event=scope.handoff.certificate.sequence,
                authentic_first_Scope_owner_returns=[list(item) for item in scope.handoff.owner_returns],
                actual_main_root_event_ids=list(range(scope.root_start + 1, scope.root_end + 1)),
                actual_Party_main_roll=engine.events[scope.root_start].amount if scope.current_actor is None else None,
                actual_current_main_roll=engine.events[scope.root_start].amount,
                all_choice_invariant="S->S_B,D; D KEEP->empty; DUEL Party or S, all72 target/dice branches->S_D,D; queued S never decreases")
        return result
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
        return None


def probe(engine):
    scope = getattr(engine, ATTRIBUTE, None)
    if type(scope) is not Scope or scope.disabled or scope.closed:
        return
    if scope.snapshot is None:
        scope.snapshot = capture(engine, scope)
        return
    certificate = completed(engine, scope.snapshot)
    if certificate is None:
        return
    scope.expected_returns = list(certificate["Duel_continuations_still_to_return"])
    engine._emit("loop_stopped", engine.turn_order[engine.turn_index], reason=REASON,
                 parent_sequence=scope.snapshot.event_start + 1, amount=len(engine.pending_effects),
                 operation=_json(certificate))
    scope.certificate_event = len(engine.events)
    scope.closed = True
    engine.pending_effects.clear()


def continuation_returned(engine, scope, trigger):
    """Actual generator return hook, also observable by real-engine tests."""
    if scope.closed:
        _need(scope.expected_returns and scope.expected_returns.pop(0) == trigger
              and len(engine.events) == scope.certificate_event and not engine.pending_effects
              and engine._duel_offered_epochs.get(_actors(engine)[3].racer_id) == _epoch(engine))
        scope.actual_returns.append(trigger)


def completion(engine, scope):
    """All original suspended generators really returned before Party resumes."""
    if scope.closed:
        _need(not scope.expected_returns and not scope.duels and not scope.drains
              and len(engine.events) == scope.certificate_event and not engine.pending_effects)


def drain_scope(method):
    @wraps(method)
    def scoped(engine, *args, **kwargs):
        scope = getattr(engine, ATTRIBUTE, None)
        owns = False
        if type(scope) is Handoff:
            # Harmless empty drains before the real main roll must not consume
            # the one-turn handoff. No action or RNG state is inspected here.
            if (len(engine.events) == len(scope.events) and tuple(engine.events) == scope.events
                    and tuple(engine.decisions) == scope.decisions and not engine.pending_effects):
                try:
                    actors = _actors(engine)
                    _base(engine, actors, current_actor=scope.current_actor, flexible_baby=scope.current_actor is not None)
                    _need(engine.racers[actors[0].racer_id if scope.current_actor is None else scope.current_actor].position == scope.start_position
                          and scope.completion_checked is True
                          and _mechanics(engine, actors, handoff=True) == scope.mechanics
                          and _past_digest(engine.events, engine.decisions) == scope.prefix_sha256
                          and tuple(sorted(engine._duel_offered_epochs.items())) == scope.raw_epoch)
                except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                    delattr(engine, ATTRIBUTE)
                yield from method(engine, *args, **kwargs)
                return
            try:
                scope = _birth_after_main(engine, _actors(engine), scope)
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                delattr(engine, ATTRIBUTE)
                yield from method(engine, *args, **kwargs)
                return
            setattr(engine, ATTRIBUTE, scope)
            owns = True
        elif scope is None:
            try:
                scope = _birth(engine, _actors(engine))
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                yield from method(engine, *args, **kwargs)
                return
            setattr(engine, ATTRIBUTE, scope)
            owns = True
        if type(scope) is not Scope:
            yield from method(engine, *args, **kwargs)
            return
        frame = Frame(scope.duels[-1].trigger if scope.duels else None, len(engine.events),
                      engine._resolution_depth, engine._physical_resolution_depth)
        scope.drains.append(frame)
        failed = False
        try:
            yield from method(engine, *args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            if scope.drains and scope.drains[-1] is frame:
                scope.drains.pop()
            if owns:
                replacement = None
                try:
                    if not failed:
                        completion(engine, scope)
                        if scope.closed and scope.phase == "before_main":
                            replacement = _make_handoff(engine, scope)
                finally:
                    if replacement is None:
                        delattr(engine, ATTRIBUTE)
                    else:
                        setattr(engine, ATTRIBUTE, replacement)
    return scoped


def duel_scope(method):
    @wraps(method)
    def scoped(engine, source, trigger_sequence):
        scope = getattr(engine, ATTRIBUTE, None)
        owns = False
        if scope is None and trigger_sequence is None:
            try:
                scope = _birth_direct(engine, _actors(engine), source, trigger_sequence)
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                scope = None
            if scope is not None:
                setattr(engine, ATTRIBUTE, scope)
                scope.root_mechanics = _mechanics(engine, _actors(engine), handoff=True)
                owns = True
        if type(scope) is not Scope:
            yield from method(engine, source, trigger_sequence)
            return
        frame = Frame(trigger_sequence, len(engine.events), engine._resolution_depth, engine._physical_resolution_depth)
        frame.source_id = source.racer_id
        scope.duels.append(frame)
        iterator = method(engine, source, trigger_sequence)
        returned = False
        try:
            while True:
                try:
                    child = next(iterator)
                except StopIteration:
                    returned = True
                    break
                frame.yielded.append(len(engine.events))
                yield child
        finally:
            iterator.close()
            if scope.duels and scope.duels[-1] is frame:
                scope.duels.pop()
            if returned:
                scope.returned.append(trigger_sequence)
                continuation_returned(engine, scope, trigger_sequence)
            if owns:
                replacement = None
                try:
                    if returned:
                        completion(engine, scope)
                        try:
                            replacement = _make_handoff(engine, scope)
                        except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                            pass
                finally:
                    if replacement is None:
                        delattr(engine, ATTRIBUTE)
                    else:
                        setattr(engine, ATTRIBUTE, replacement)
    return scoped

"""Authenticate one dynamic Copy/Centaur/Star FIFO; never execute or erase jobs.

On Wild, P=Centaur12, B=Baby5 and C=Copy4. S ranges12..15. C may
legally borrow P at12; its already qualified Scoocher tasks still execute.
For T=#S+2#C, U=T+I[S=13] never decreases over any legal task/choice.
At the first clean Root14/CopyS capture with T>=2 the FIFO can never drain. A single
S at12 while C borrows Centaur is a legal finite counterexample and excluded.
"""

from collections import deque
from dataclasses import asdict, dataclass, fields
import json

from .copy_fish_baby_cycle import _ENGINE_FIELDS
from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .fish_baby_star_cycle import _event
from .racer import Ability, Racer
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS


REASON = "centaur_copy_baby_star_choice_fifo_cycle_proved"
_ROLES = (Ability.CENTAUR, Ability.COPY_CAT, Ability.HUGE_BABY, Ability.SCOOCHER)
_RACER_FIELDS = frozenset(field.name for field in fields(Racer))
_EVENT_FIELDS = frozenset(field.name for field in fields(RaceEvent))
_PENDING_FIELDS = frozenset(field.name for field in fields(PendingEffect))
_CONTROL_CONTAINERS = {
    "_copy_locks": set, "pending_effects": deque, "events": list, "decisions": list,
    "turns": list, "turn_order": list, "finish_order": list, "racers": dict,
    "policies": dict, "policy_metadata": dict,
}


class _Reject(Exception):
    pass


def _actors(engine):
    if len(engine.racers) != 4:
        return None
    roles = {actor.ability: actor for actor in engine.racers.values()}
    return tuple(roles[role] for role in _ROLES) if set(roles) == set(_ROLES) else None


def _successors(control, job, notification_order="SC"):
    """The complete finite control algebra, independent of any real policy."""
    phase, mode = control
    if (phase not in (12, 13, 14, 15) or mode not in ("S", "P")
            or phase != 12 and mode != "S" or notification_order not in ("SC", "CS")):
        return ()
    if job == "S":
        word = "" if mode == "P" else "C"
        after = 12 if phase == 15 else phase + 1
        modes = ("S", "P") if after == 12 else ("S",)
        return tuple(((after, new_mode), word, int(phase == 12)) for new_mode in modes)
    if job == "C":
        word = "SS" if mode == "P" else "S" + notification_order
        modes = ("S", "P") if phase == 12 else ("S",)
        return tuple(((phase, new_mode), word, 0) for new_mode in modes)
    return ()


def _weight(queue, s, c):
    return sum(1 if job.racer_id == s else 2 if job.racer_id == c else 0 for job in queue)


def _snapshot(entries, players):
    totals = dict.fromkeys(players, 0)
    races = {race: dict.fromkeys(players, 0) for race in range(1, 5)}
    for index, entry in enumerate(entries, 1):
        if (entry.sequence != index or type(entry.sequence) is not int or entry.player_id not in totals
                or type(entry.delta) is not int or entry.race_index not in races):
            raise _Reject
        totals[entry.player_id] += entry.delta
        races[entry.race_index][entry.player_id] += entry.delta
    return ScoreSnapshot(tuple(players), tuple(totals.items()),
                         tuple((race, tuple(values.items())) for race, values in races.items()), tuple(entries))


def _trace(snapshot):
    return {"schema_version": 1, "player_ids": list(snapshot.player_ids),
            "totals": {str(rid): total for rid, total in snapshot.totals},
            "race_totals": {str(race): {str(rid): total for rid, total in values}
                            for race, values in snapshot.race_totals},
            "entries": json.loads(json.dumps([asdict(entry) for entry in snapshot.entries]))}


def _ledger_cut(engine, boundary):
    current = engine.score_ledger.snapshot()
    if (set(current.player_ids) != set(engine.racers) or type(engine._score_start) is not int
            or not 0 <= engine._score_start <= len(current.entries)
            or _snapshot(current.entries, current.player_ids) != current
            or _trace(_snapshot(current.entries[:engine._score_start], current.player_ids)) != engine.initial_score_ledger):
        raise _Reject
    count = engine._score_start
    seen_future = False
    for entry in current.entries[engine._score_start:]:
        if (entry.race_index != engine.race_index or type(entry.event_sequence) is not int
                or not 1 <= entry.event_sequence <= len(engine.events)):
            raise _Reject
        if entry.event_sequence <= boundary:
            if seen_future:
                raise _Reject
            count += 1
        else:
            seen_future = True
    base = _snapshot(current.entries[:count], current.player_ids)
    event_ids = [event.sequence for event in engine.events[:boundary] if event.kind == "score"]
    if event_ids != [entry.event_sequence for entry in current.entries[engine._score_start:count]]:
        raise _Reject
    balances = dict(_snapshot(current.entries[:engine._score_start], current.player_ids).totals)
    for entry in current.entries[engine._score_start:count]:
        event = engine.events[entry.event_sequence - 1]
        if (entry.delta != 1 or entry.source != "track" or entry.reason != "star"
                or entry.phase != "during_race" or type(event.parent_sequence) is not int
                or not 1 <= event.parent_sequence < event.sequence
                or not _event(event, "score", entry.player_id, before=balances[entry.player_id],
                              after=balances[entry.player_id] + 1, amount=1,
                              parent_sequence=event.parent_sequence, reason="star")):
            raise _Reject
        stop = engine.events[event.parent_sequence - 1]
        if stop.kind != "stop" or stop.racer_id != entry.player_id or stop.after not in engine.track.star_positions:
            raise _Reject
        balances[entry.player_id] += 1
    if dict(base.totals) != balances:
        raise _Reject
    return base


def _score_return(engine, old, reader):
    current = engine.score_ledger.snapshot()
    if current.player_ids != old.player_ids or current.entries[:len(old.entries)] != old.entries:
        return False
    added = current.entries[len(old.entries):]
    if len(added) != len(reader.stars):
        return False
    for index, (entry, event) in enumerate(zip(added, reader.stars), len(old.entries) + 1):
        if ((entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source, entry.reason,
             entry.event_sequence, entry.phase) != (index, reader.s, engine.race_index, 1, "track", "star",
                                                   event.sequence, "during_race")):
            return False
    return (_snapshot(current.entries, current.player_ids) == current
            and current.totals == tuple((rid, reader.balances[rid]) for rid, _ in old.totals))


class _Reader:
    """Consume exact existing events/choices; this object never calls an action."""
    def __init__(self, engine, actors, start, phase, mode, score, decision, batch):
        self.engine, self.actors = engine, actors
        self.p, self.c, self.b, self.s = (actor.racer_id for actor in actors)
        self.events = engine.events
        self.offset, self.decision, self.batch = start, decision, batch
        self.positions = {self.p: 12, self.c: 4, self.b: 5, self.s: phase}
        self.mode = mode
        self.target = self.s if mode == "S" else self.p
        self.balances = dict(score.totals)
        self.score = score
        self.order = tuple(engine._clockwise_order())
        self.children, self.stars, self.choice_ids, self.change_ids = [], [], [], []
        self.executed, self.s_jobs, self.c_jobs = [], 0, 0
        self.potential_changes = []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def queued(self, actor, source, power):
        event = self.take("effect_queued", actor, source_id=source, ability=Ability.SCOOCHER.value,
                          parent_sequence=power.sequence)
        self.children.append(PendingEffect(actor, source, power.sequence, event.sequence))

    def notify(self, source, power):
        for rid in self.order:
            if rid == self.s and source != self.s or rid == self.c and source != self.c and self.mode == "S":
                self.queued(rid, source, power)

    def observation(self):
        by_id = {actor.racer_id: actor for actor in self.actors}
        public = []
        for rid in self.engine.turn_order:
            actor = by_id[rid]
            ability = (Ability.SCOOCHER if self.mode == "S" else Ability.CENTAUR) if rid == self.c else actor.ability
            public.append(PublicRacer(rid, actor.name, self.positions[rid], "active", ability.value,
                                     actor.tripped, None, self.target if rid == self.c else None))
        return Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            WILD_WILDS.name, WILD_WILDS.finish_position, WILD_WILDS.second_corner_position,
            tuple(sorted(WILD_WILDS.trip_positions)), tuple(public), (), None, None,
            score_totals=tuple((rid, self.balances[rid]) for rid, _ in self.score.totals))

    def refresh(self):
        targets = tuple(rid for rid in self.order if rid in {self.p, self.s}
                        and self.positions[rid] == max(self.positions.values()))
        old = self.target, self.mode
        if len(targets) == 2:
            legal = (Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets))
            if self.decision >= len(self.engine.decisions):
                raise _Reject
            record = self.engine.decisions[self.decision]
            public = record.request.observation
            if (type(record.request) is not DecisionRequest or type(public) is not Observation
                    or any(type(getattr(public, name)) is not int for name in (
                        "current_player_id", "turn_number", "finish_position", "second_corner_position",
                        "magician_rerolls_remaining", "dicemonger_rerolls_remaining"))
                    or any(type(value) is not int for pair in public.score_totals for value in pair)
                    or any(type(value) is not int for value in public.trip_positions)
                    or any(type(actor) is not PublicRacer or type(actor.racer_id) is not int
                           or type(actor.position) is not int or type(actor.tripped) is not bool
                           or actor.copied_target is not None and type(actor.copied_target) is not int
                           for actor in public.racers)
                    or any(type(action) is not Action or not isinstance(action.kind, ActionKind)
                           or any(value is not None and type(value) is not int
                                  for value in (action.source_id, action.target_id))
                           for action in (*record.request.legal_actions, record.action))):
                raise _Reject
            request = DecisionRequest(self.decision + 1, self.c, DecisionWindow.COPY_CHOICE,
                                      self.observation(), legal)
            if (record.request != request or record.action not in legal or type(record.request.player_id) is not int
                    or type(record.request.decision_id) is not int or type(record.event_sequence) is not int):
                raise _Reject
            event = self.take("decision", self.c, source_id=record.action.source_id,
                              target_id=record.action.target_id, action_kind=record.action.kind.value,
                              decision_id=request.decision_id)
            if record.event_sequence != event.sequence:
                raise _Reject
            self.decision += 1
            self.choice_ids.append(event.sequence)
            if record.action.kind != ActionKind.KEEP:
                self.target = record.action.target_id
        elif targets == (self.s,):
            self.target = self.s
        else:
            raise _Reject
        self.mode = "S" if self.target == self.s else "P"
        if old != (self.target, self.mode):
            event = self.take("copy_changed", self.c, source_id=self.c, target_id=self.target,
                              ability=(Ability.SCOOCHER if self.mode == "S" else Ability.CENTAUR).value)
            self.change_ids.append(event.sequence)

    def root(self, parent, amount):
        """The admitted real C main-move family: start5-n -> Baby5 ->4."""
        self.children = []
        self.batch += 1
        self.take("move", self.c, before=5 - amount, after=5, amount=amount,
                  movement_kind="main", parent_sequence=parent.sequence, arrival_batch_id=self.batch)
        self.positions[self.c] = 5
        baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                         parent_sequence=parent.sequence, arrival_batch_id=self.batch)
        self.notify(self.b, baby)
        self.batch += 1
        self.take("stop", self.c, before=5, after=4, parent_sequence=baby.sequence, arrival_batch_id=self.batch)
        self.positions[self.c] = 4
        # Original S14 is the sole leader both while the main power is locked
        # and after the real main movement releases Copy's instance lock.
        return tuple(self.children)

    def job(self, pending):
        before = self.positions[self.s], self.mode
        actor = pending.racer_id
        if actor not in {self.s, self.c}:
            raise _Reject
        label = "S" if actor == self.s else "C"
        self.children = []
        stars_before = len(self.stars)
        power = self.take("ability_effect", actor, source_id=actor, ability=Ability.SCOOCHER.value,
                          amount=1, parent_sequence=pending.queued_sequence)
        # A qualified old C instance still emits its stored power after C
        # changes borrowing. Watchers use the current actual abilities.
        self.notify(actor, power)
        origin = self.positions[actor]
        self.batch += 1
        move = self.take("move", actor, source_id=actor, ability=Ability.SCOOCHER.value,
                         before=origin, after=origin + 1, amount=1, movement_kind="ability",
                         parent_sequence=power.sequence, arrival_batch_id=self.batch)
        self.positions[actor] = origin + 1
        if actor == self.c:
            baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                             parent_sequence=power.sequence, arrival_batch_id=self.batch)
            self.notify(self.b, baby)
            self.batch += 1
            self.take("stop", self.c, before=5, after=4, parent_sequence=baby.sequence, arrival_batch_id=self.batch)
            self.positions[self.c] = 4
            self.c_jobs += 1
        else:
            stop = self.take("stop", self.s, before=origin, after=origin + 1,
                             parent_sequence=move.sequence, arrival_batch_id=self.batch)
            if origin == 12:
                balance = self.balances[self.s]
                event = self.take("score", self.s, before=balance, after=balance + 1, amount=1,
                                  parent_sequence=stop.sequence, reason="star")
                self.balances[self.s] += 1
                self.stars.append(event)
            elif origin == 15:
                self.batch += 1
                arrow = self.take("move", self.s, before=16, after=12, amount=-4, movement_kind="track",
                                  parent_sequence=stop.sequence, arrival_batch_id=self.batch)
                self.take("stop", self.s, before=16, after=12, parent_sequence=arrow.sequence,
                          arrival_batch_id=self.batch)
                self.positions[self.s] = 12
            self.s_jobs += 1
        self.refresh()
        word = "".join("S" if job.racer_id == self.s else "C" for job in self.children)
        order = "".join("S" if rid == self.s else "C" for rid in self.order if rid in {self.s, self.c})
        after = (self.positions[self.s], self.mode), word, len(self.stars) - stars_before
        if after not in _successors(before, label, order):
            raise _Reject
        self.executed.append(pending.queued_sequence)
        return tuple(self.children)


def _pending_shape(engine, queue, actors, *, unconsumed):
    p, c, b, s = (actor.racer_id for actor in actors)
    if not queue or any(type(job) is not PendingEffect or set(vars(job)) != _PENDING_FIELDS for job in queue):
        return False
    ids = [job.queued_sequence for job in queue]
    if any(type(value) is not int for value in ids) or ids != sorted(set(ids)):
        return False
    consumed = {event.parent_sequence for event in engine.events if event.kind in {"ability_effect", "effect_cancelled"}}
    if unconsumed and consumed.intersection(ids):
        return False
    for job in queue:
        if (any(type(v) is not int for v in (job.racer_id, job.source_id, job.steps, job.trigger_sequence))
                or job.racer_id not in {s, c} or job.source_id not in {s, c, b}
                or (job.kind, job.steps, job.ability, job.target_id) != ("move", 1, Ability.SCOOCHER.value, None)
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)):
            return False
        trigger, queued = engine.events[job.trigger_sequence - 1], engine.events[job.queued_sequence - 1]
        if (type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                or not _event(queued, "effect_queued", job.racer_id, source_id=job.source_id,
                              ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)):
            return False
        if job.source_id == b:
            if (type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1
                    or not _event(trigger, "ability_effect", b, source_id=b, ability=Ability.HUGE_BABY.value,
                                  parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)):
                return False
        elif not _event(trigger, "ability_effect", job.source_id, source_id=job.source_id,
                        ability=Ability.SCOOCHER.value, amount=1, parent_sequence=trigger.parent_sequence):
            return False
    return True


def _history_before(engine, actors, boundary):
    """Read the actual earlier position/Copy/trip records, without rule execution."""
    initial = {row["racer_id"]: row for row in engine.initial_racers}
    if len(initial) != 4 or set(initial) != set(engine.racers):
        raise _Reject
    for actor, role in zip(actors, _ROLES):
        if initial[actor.racer_id]["ability"] != role.value or initial[actor.racer_id]["inherited_ability"] is not None:
            raise _Reject
    positions = {rid: row["position"] for rid, row in initial.items()}
    trips = {rid: row["tripped"] for rid, row in initial.items()}
    c = actors[1].racer_id
    target = initial[c]["copied_target"]
    ability = initial[c]["copied_ability"]
    for event in engine.events[:boundary]:
        if event.kind in {"move", "warp"}:
            if event.before != positions[event.racer_id] or type(event.after) is not int or not 0 <= event.after < 30:
                raise _Reject
            positions[event.racer_id] = event.after
        elif event.kind == "stop":
            if positions[event.racer_id] not in {event.before, event.after} or type(event.after) is not int or not 0 <= event.after < 30:
                raise _Reject
            positions[event.racer_id] = event.after
        elif event.kind == "copy_changed":
            if event.racer_id != c:
                raise _Reject
            expected = engine.racers[event.target_id].ability.value if event.target_id is not None else None
            if (event.target_id == c or not _event(event, "copy_changed", c, source_id=c,
                                                  target_id=event.target_id, ability=expected)):
                raise _Reject
            target, ability = event.target_id, event.ability
        elif event.kind == "trip":
            trips[event.racer_id] = True
        elif event.kind == "recover":
            trips[event.racer_id] = False
        elif event.kind in {"finish", "eliminate"}:
            raise _Reject
    return positions, trips, target, ability


def _birth(engine, actors):
    """Authenticate every current job by parsing its real complete root suffix."""
    p, c, b, s = (actor.racer_id for actor in actors)
    for parent in reversed(engine.events):
        if parent.kind != "main_move_resolved" or parent.racer_id != c:
            continue
        try:
            amount = parent.after
            if (type(amount) is not int or not 1 <= amount <= 5 or parent.before != amount
                    or parent.parent_sequence != parent.sequence - 1
                    or not _event(parent, "main_move_resolved", c, before=amount, after=amount,
                                  parent_sequence=parent.sequence - 1)):
                continue
            locked, roll = engine.events[parent.sequence - 2], engine.events[parent.sequence - 3]
            if (not _event(locked, "roll_locked", c, amount=amount, parent_sequence=roll.sequence)
                    or not _event(roll, "roll", c, amount=amount)
                    or locked.sequence != parent.sequence - 1 or roll.sequence != parent.sequence - 2):
                continue
            positions, trips, target, ability = _history_before(engine, actors, parent.sequence)
            if (positions != {p: 12, c: 5 - amount, b: 5, s: 14}
                    or target != s or ability != Ability.SCOOCHER.value or trips[c] is not False
                    or any(trips[actor.racer_id] != actor.tripped for actor in actors)):
                continue
            direct = engine.events[parent.sequence]
            if direct.kind != "move" or type(direct.arrival_batch_id) is not int or direct.arrival_batch_id < 1:
                continue
            score = _ledger_cut(engine, parent.sequence)
            decision = sum(record.event_sequence <= parent.sequence for record in engine.decisions)
            reader = _Reader(engine, actors, parent.sequence, 14, "S", score, decision, direct.arrival_batch_id - 1)
            queue = deque(reader.root(parent, amount))
            root_ids = [job.queued_sequence for job in queue]
            while reader.offset < len(engine.events):
                if not queue:
                    raise _Reject
                queue.extend(reader.job(queue.popleft()))
            if (tuple(queue) != tuple(engine.pending_effects) or reader.decision != len(engine.decisions)
                    or reader.positions != {actor.racer_id: actor.position for actor in actors}
                    or reader.target != actors[1].copied_target
                    or (Ability.SCOOCHER if reader.mode == "S" else Ability.CENTAUR) != actors[1].copied_ability
                    or reader.batch != engine._arrival_counter or not _score_return(engine, score, reader)):
                continue
            return {"root_main_sequence": parent.sequence, "root_roll_sequence": roll.sequence,
                    "root_die": amount, "root_Copy_start": 5 - amount, "root_pending_ids": root_ids,
                    "root_source_S_phase": 14}
        except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
            continue
    return None


def key(engine):
    try:
        if (any(type(getattr(engine, name, None)) is not expected
                for name, expected in _CONTROL_CONTAINERS.items())
                or type(engine.race_index) is not int or not 1 <= engine.race_index <= 4
                or any(type(rid) is not int for rid in engine.turn_order)
                or type(getattr(engine, "_movement_depth", 0)) is not int
                or set(vars(engine)) - _ENGINE_FIELDS or engine.track != WILD_WILDS
                or engine._preparing is not False or engine._turn_in_progress is not True
                or engine.termination_reason is not None or engine.finish_order
                or engine._copy_locks or engine._refreshing_copies is not False
                or getattr(engine, "_movement_depth", 0) != 0
                or getattr(engine, "_inert_duel_romantic", None) is not None
                or getattr(engine, "_duel_offered_epochs", {})
                or engine.next_turn_racer_id is not None or engine._genius_prediction is not None
                or any(type(getattr(engine, name, None)) is not int or getattr(engine, name) < 1
                       for name in ("_resolution_depth", "_physical_resolution_depth", "_arrival_counter"))
                or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
                or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
            return None
        actors = _actors(engine)
        if actors is None:
            return None
        p, c, b, s = actors
        if (engine.turn_order[engine.turn_index] != c.racer_id
                or (p.position, c.position, b.position, s.position) != (12, 4, 5, 14)
                or c.copied_target != s.racer_id or c.copied_ability != Ability.SCOOCHER or c.tripped is not False):
            return None
        for actor in actors:
            if (type(actor) is not Racer or set(vars(actor)) != _RACER_FIELDS
                    or type(actor.racer_id) is not int or engine.racers.get(actor.racer_id) is not actor
                    or type(actor.position) is not int or actor.status != RacerStatus.ACTIVE
                    or actor.pre_race_resolved is not True or actor.inherited_ability is not None
                    or actor.finish_place is not None or actor.finish_places or actor.prediction is not None
                    or type(actor.tripped) is not bool or type(actor.turns_taken) is not int or actor.turns_taken < 0
                    or actor is not c and (actor.copied_target is not None or actor.copied_ability is not None)):
                return None
        if any(type(event) is not RaceEvent or set(vars(event)) != _EVENT_FIELDS
               or type(event.sequence) is not int or event.sequence != index + 1
               for index, event in enumerate(engine.events)):
            return None
        pending = tuple(engine.pending_effects)
        if not _pending_shape(engine, pending, actors, unconsumed=True) or _weight(pending, s.racer_id, c.racer_id) < 2:
            return None
        birth = _birth(engine, actors)
        if birth is None:
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                              "policy_metadata": engine.policy_metadata}, sort_keys=True, separators=(",", ":"))
        state = (engine.track, roster, tuple(engine.turns), engine.turn_index, tuple(engine.turn_order),
                 engine.race_index, tuple(engine.remaining_abilities), tuple(engine.initial_remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._preparing, engine._turn_in_progress,
                 engine._resolution_depth, engine._physical_resolution_depth, initial, engine._score_start,
                 engine.initial_code_fingerprint, engine.max_events, getattr(engine, "max_action_events", None),
                 engine.seed, tuple(sorted(engine.policies)), tuple(sorted(vars(engine))), tuple(sorted(birth.items())))
        # Root queue IDs are ordered immutable provenance, not mutable state.
        state = (*state[:-1], json.dumps(birth, sort_keys=True, separators=(",", ":")))
        hash(state)
        return state
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
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
        raise ValueError("not a certified dynamic Centaur/Copy/Star FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine.score_ledger.snapshot(), engine._arrival_counter,
                    tuple(engine.events), tuple(engine.decisions))


def completed(engine, snapshot):
    try:
        if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
                or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix):
            return False
        actors = _actors(engine)
        p, c, b, s = (actor.racer_id for actor in actors)
        if not _pending_shape(engine, snapshot.original_pending, actors, unconsumed=False):
            return False
        reader = _Reader(engine, actors, snapshot.event_start, 14, "S", snapshot.score_before,
                         snapshot.decision_start, snapshot.arrival_counter)
        queue = deque(snapshot.original_pending)
        before_t = _weight(queue, s, c)
        if before_t < 2:
            return False
        while reader.offset < len(engine.events):
            if not queue:
                return False
            before_u = _weight(queue, s, c) + int(reader.positions[s] == 13)
            queue.extend(reader.job(queue.popleft()))
            after_u = _weight(queue, s, c) + int(reader.positions[s] == 13)
            if after_u < before_u or after_u < 2:
                return False
            reader.potential_changes.append((before_u, after_u))
        original = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in queue]
        if (reader.positions[s] != 14 or reader.mode != "S" or reader.s_jobs < 4 or reader.s_jobs % 4
                or reader.c_jobs < 1 or [identity for identity in reader.executed if identity in original] != original
                or len(reader.executed) != len(set(reader.executed)) or not queue or set(original).intersection(fresh)
                or any(identity <= snapshot.event_start for identity in fresh)
                or tuple(queue) != tuple(engine.pending_effects) or reader.batch != engine._arrival_counter
                or reader.decision != len(engine.decisions) or not _score_return(engine, snapshot.score_before, reader)):
            return False
        birth = _birth(engine, actors)
        return {"capture_event_count": snapshot.event_start, "original_pending_ids": original,
                "executed_pending_ids": reader.executed, "new_pending_ids": fresh,
                "queue_before": len(original), "queue_after": len(fresh),
                "T_before": before_t, "T_after": _weight(queue, s, c),
                "U_before": before_t, "U_after": _weight(queue, s, c),
                "actual_U_step_pairs": reader.potential_changes,
                "scoocher_jobs": reader.s_jobs, "copy_jobs": reader.c_jobs,
                "scoocher_complete_orbits": reader.s_jobs // 4,
                "copy_decision_event_ids": reader.choice_ids, "copy_changed_event_ids": reader.change_ids,
                "score_event_ids": [event.sequence for event in reader.stars],
                "star_point_increments": {str(rid): reader.balances[rid] - total for rid, total in snapshot.score_before.totals},
                "root_source_evidence": birth, "all_original_jobs_executed": True,
                "mechanical_state_returned": True, "only_fresh_pending_descendants_stopped": True,
                "all_real_Star_entries_retained": True,
                "proof": "all actual Copy choices; qualified old C instances survive borrowing changes; U=#S+2#C+I[S13] never decreases; first eligible Root14 capture, exact FIFO, full phase14 return and real Star ledger"}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False

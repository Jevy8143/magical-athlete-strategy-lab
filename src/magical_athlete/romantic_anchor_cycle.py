"""An all-choice, zero-score Romantic loop with a stationary anchor on Wild22.

R/P remain22; an original Copy stays12/14, borrowing either R or P. Every
qualified R job returns22->24->22 and replaces itself, optionally queuing C.
An already-qualified C Romantic job still moves even after Copy chooses P;
with spectator17 its12->14 or14->16->12 journey has no child. The passive domains
below were checked for BOTH stationary P and Copy's real borrowed movement.

Each clockwise profile has a naturally closed16-configuration FIFO family,
including all three real Copy actions at every boundary. This is not a
queue budget or a claim that all policies choose the same mechanical state.
Every captured original job must run, all actual choices must authenticate,
and full mechanical state/ledger must return before only fresh jobs stop.

The separate native Alchemist22/ThirdWheel12 domain adds real R(+C) births
on Copy's return12. Its FIFO is unbounded: a production invariant, rather
than the spectator17 finite table, proves a persistent R token for all choices.
"""

from collections import Counter, deque
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .static_inheritance import static_inheritance_evidence
from .track import WILD_WILDS


REASON = "romantic_anchor_copy_queue_cycle_proved"
THIRD_WHEEL12_PROFILE = "native_alchemist22_third_wheel12"


def _third_wheel12(actors):
    """An explicit extra native domain, not a general spectator relaxation."""
    r, c, p, s = actors
    return (c.ability == Ability.COPY_CAT and c.inherited_ability is None
            and p.ability == Ability.ALCHEMIST and p.position == 22
            and s.ability == Ability.THIRD_WHEEL and s.position == 12
            and all(actor.inherited_ability is None and actor.prediction is None for actor in actors))


def _renewing_successor(state, action, profile):
    """The queue word is unbounded; its R-token count never decreases.

    R returns to stationary Alchemist22 and qualifies R (+C if Copy is R).
    C12->14 has no child; C14->16->12 meets ThirdWheel12 and qualifies the
    same R (+C). Births precede the actual subsequent Copy choice. Alchemist
    only modifies a main roll and ThirdWheel only acts at its own turn start;
    neither has a body in this pure, already-qualified FIFO.
    """
    position, copied_r, word = state
    if (position not in (12, 14) or type(copied_r) is not bool or not word
            or any(token not in "RC" for token in word) or "R" not in word
            or profile not in {"RC", "CR"} or action not in {"keep", "copy_p", "copy_r"}):
        raise ValueError("outside renewing R/C grammar")
    head, tail = word[0], word[1:]
    children = (profile if copied_r else "R") if head == "R" or position == 14 else ""
    following = copied_r if action == "keep" else action == "copy_r"
    return (position if head == "R" else 26 - position, following, tail + children)


@lru_cache(maxsize=2)
def _renewing_proof(profile):
    if profile not in {"RC", "CR"}:
        raise ValueError("unknown renewing notification order")
    edges = []
    for position in (12, 14):
        for copied_r in (False, True):
            for head in "RC":
                children = (profile if copied_r else "R") if head == "R" or position == 14 else ""
                if children.count("R") < int(head == "R"):
                    raise ValueError("a real job could consume the last R token")
                for action in ("keep", "copy_p", "copy_r"):
                    # The untouched arbitrary tail is deliberately not enumerated.
                    # These are 24 control/production edges, not a bounded queue.
                    edges.append((position, copied_r, head, action, children))
    digest = hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest()
    return tuple(edges), digest

# These are deliberately explicit, independently reviewed domains. The own-
# movement listeners MOUTH/CENTAUR/LEAPTOAD have no other racer to encounter
# on Copy's12/14/16 path; STICKLER sees no finish overrun. The remaining26
# powers have only turn/roll/main-move windows in this prefix. No inherited
# card, Scoocher, active Copy spectator or future new power enters implicitly.
ANCHOR_ABILITIES = frozenset({
    Ability.SKIPPER, Ability.HECKLER, Ability.INCHWORM, Ability.LACKEY,
    Ability.ALCHEMIST, Ability.BLIMP, Ability.CENTAUR, Ability.CHEERLEADER,
    Ability.COACH, Ability.DICEMONGER, Ability.FLIP_FLOP, Ability.GENIUS,
    Ability.GUNK, Ability.HARE, Ability.HYPNOTIST, Ability.LEAPTOAD,
    Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN, Ability.MASTERMIND,
    Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.STICKLER, Ability.THIRD_WHEEL,
})

# At17 these original spectators are outside BOTH movement paths. In
# particular Banana/Baba/Baby/Fish cannot hear a relevant arrival/passing.
SPECTATOR_ABILITIES = frozenset({
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
    Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER,
    Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE, Ability.HECKLER,
    Ability.HUGE_BABY, Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY,
    Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN,
    Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL,
})


def _successor(state, action, profile):
    position, copied_r, word = state
    actor, tail = word[0], word[1:]
    children = (profile if copied_r else "R") if actor == "R" else ""
    following = copied_r if action == "keep" else action == "copy_r"
    return (position if actor == "R" else 26 - position, following, tail + children)


@lru_cache(maxsize=2)
def _finite_proof(profile):
    if profile not in {"RC", "CR"}:
        raise ValueError("unknown Romantic/Copy clockwise profile")
    # R and RC are safe seed words; include both so an authenticated older
    # Copy job behind R remains eligible in either clockwise profile.
    states = {(position, copied_r, word) for position in (12, 14)
              for copied_r in (False, True) for word in ("R", "RC")}
    todo, edges = deque(sorted(states)), []
    while todo:
        state = todo.popleft()
        for action in ("keep", "copy_p", "copy_r"):
            target = _successor(state, action, profile)
            if not target[2] or target[2].count("R") != 1:
                raise ValueError("anchor FIFO family lost its persistent Romantic")
            edges.append((state, action, target))
            if target not in states:
                states.add(target)
                todo.append(target)
    # Natural exhaustion, not a length/node/time enumeration cutoff.
    if len(states) != 16 or len(edges) != 48:
        raise ValueError("anchor FIFO all-choice family changed")
    for state in states:
        for action in ("keep", "copy_p", "copy_r"):
            if _successor(state, action, profile) not in states:
                raise ValueError("anchor FIFO family is not closed")
    table = {"profile": profile, "states": sorted(states), "edges": sorted(edges)}
    digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return frozenset(states), tuple(sorted(edges)), digest


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _copy_source(engine, actor):
    """Only a sole original Copy or an authenticated, completed static Copy."""
    if actor.ability == Ability.COPY_CAT:
        return ()
    if actor.ability not in {Ability.EGG, Ability.TWIN} or actor.inherited_ability != Ability.COPY_CAT:
        return None
    evidence = static_inheritance_evidence(engine, actor, Ability.COPY_CAT)
    if not evidence:
        return None
    binding = json.loads(evidence[0])
    grants = [item["sequence"] for item in binding["inherit_events"]
              if item["racer_id"] == actor.racer_id and item["ability"] == Ability.COPY_CAT.value]
    if len(grants) != 1:
        return None
    grant = grants[0]
    if any(event.racer_id == actor.racer_id and event.sequence <= grant
           and event.kind in {"ability_effect", "copy_changed"} for event in engine.events):
        return None
    return grant, evidence


def _copy_metadata(actor, source):
    return {"racer_id": actor.racer_id, "original_ability": actor.ability.value,
            "inherited_ability": Ability.COPY_CAT.value, "copy_grant_sequence": source[0],
            "acquisition": source[1]}


def _actors(engine):
    actors = tuple(engine.racers.values())
    if (len(actors) != 4 or len({actor.ability for actor in actors}) != 4
            or any(not isinstance(actor.ability, Ability) for actor in actors)):
        return None
    romantic = next((actor for actor in actors if actor.ability == Ability.ROMANTIC), None)
    copies = [actor for actor in actors if engine._is_copy(actor)]
    if romantic is None or len(copies) != 1:
        return None
    copy = copies[0]
    if copy.ability not in {Ability.COPY_CAT, Ability.EGG, Ability.TWIN}:
        return None
    others = [actor for actor in actors if actor is not romantic and actor is not copy]
    anchors = [actor for actor in others if actor.position == 22 and actor.ability in ANCHOR_ABILITIES]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    spectator = next(actor for actor in others if actor is not anchor)
    if ((spectator.position != 17 or spectator.ability not in SPECTATOR_ABILITIES)
            and not _third_wheel12((romantic, copy, anchor, spectator))):
        return None
    return romantic, copy, anchor, spectator


def _queued(event, actor, trigger):
    if not _event(event, "effect_queued", actor, source_id=actor, target_id=actor,
                  ability=Ability.ROMANTIC.value, parent_sequence=trigger.sequence):
        return None
    return PendingEffect(actor, actor, trigger.sequence, event.sequence, "move", 2, Ability.ROMANTIC.value, actor)


def _queue_valid(engine, pending, actors, *, require_unconsumed=True):
    """Authenticate real historical single-arriver qualifications, not flags."""
    romantic, copy, _, _ = actors
    source = _copy_source(engine, copy)
    if source is None:
        return False
    grant = source[0] if source else None
    allowed_inherits = ({item["sequence"]: RaceEvent(**item) for item in json.loads(source[1][0])["inherit_events"]
                        if item["racer_id"] == copy.racer_id} if source else {})
    ids = [job.queued_sequence for job in pending if isinstance(job, PendingEffect)]
    if not pending or len(ids) != len(pending) or ids != sorted(set(ids)):
        return False
    needed = {job.trigger_sequence for job in pending}
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    if set(initial) != set(engine.racers):
        return False
    positions = {rid: item["position"] for rid, item in initial.items()}
    power = initial[copy.racer_id]["copied_ability"]
    target = initial[copy.racer_id]["copied_target"]
    at_trigger, consumed, batches = {}, set(), Counter()
    for event in engine.events:
        if event.kind in {"move", "warp", "stop"}:
            positions[event.racer_id] = event.after
        elif event.kind == "copy_changed" and event.racer_id == copy.racer_id:
            target, power = event.target_id, event.ability
        elif event.kind == "inherit":
            if event != allowed_inherits.get(event.sequence):
                return False
        elif event.kind in {"finish", "eliminate", "race_end"}:
            return False
        if event.kind in {"ability_effect", "effect_cancelled"}:
            consumed.add(event.parent_sequence)
        if event.kind == "stop":
            batches[event.arrival_batch_id] += 1
        if event.sequence in needed:
            at_trigger[event.sequence] = dict(positions), target, power
    if (positions != {rid: actor.position for rid, actor in engine.racers.items()}
            or (target, power) != (copy.copied_target, copy.copied_ability)):
        return False
    for job in pending:
        if (any(type(value) is not int for value in (job.racer_id, job.source_id, job.target_id,
                                                     job.steps, job.trigger_sequence, job.queued_sequence))
                or job.racer_id not in {romantic.racer_id, copy.racer_id}
                or (job.source_id, job.kind, job.steps, job.ability, job.target_id) != (
                    job.racer_id, "move", 2, Ability.ROMANTIC.value, job.racer_id)
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)
                or require_unconsumed and job.queued_sequence in consumed
                or grant is not None and job.trigger_sequence <= grant):
            return False
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        previous, old_target, old_power = at_trigger[job.trigger_sequence]
        if (trigger.kind != "stop" or trigger.racer_id not in engine.racers
                or type(trigger.before) is not int or type(trigger.after) is not int
                or not 0 <= trigger.after < engine.track.finish_position
                or type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                or type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1
                or not _event(trigger, "stop", trigger.racer_id, before=trigger.before, after=trigger.after,
                              parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)
                or batches[trigger.arrival_batch_id] != 1
                or previous[trigger.racer_id] != trigger.after
                or sum(position == trigger.after for position in previous.values()) != 2
                or _queued(queued, job.racer_id, trigger) != job
                or job.racer_id == copy.racer_id
                and (old_target, old_power) != (romantic.racer_id, Ability.ROMANTIC.value)):
            return False
    return True


def key(engine):
    if (engine.track != WILD_WILDS or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0
            or type(getattr(engine, "_resolution_depth", 0)) is not int
            or getattr(engine, "_resolution_depth", 0) < 1
            or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    romantic, copy, anchor, _ = actors
    if (romantic.position != 22 or copy.position not in (12, 14)
            or copy.copied_target not in {romantic.racer_id, anchor.racer_id}
            or copy.copied_ability != (Ability.ROMANTIC if copy.copied_target == romantic.racer_id else anchor.ability)):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE
                or actor.inherited_ability is not None and not (actor is copy
                    and actor.ability in {Ability.EGG, Ability.TWIN} and actor.inherited_ability == Ability.COPY_CAT)
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or type(actor.tripped) is not bool or actor.pre_race_resolved is not True
                or type(actor.turns_taken) is not int or actor.turns_taken < 0
                or actor.finish_place is not None or actor.finish_places
                or actor.prediction is not None and (type(actor.prediction) is not int or actor.prediction not in engine.racers)
                or actor is not copy and (actor.copied_target is not None or actor.copied_ability is not None)):
            return None
    profile = "".join("R" if rid == romantic.racer_id else "C" for rid in engine._clockwise_order()
                      if rid in {romantic.racer_id, copy.racer_id})
    try:
        pending = tuple(engine.pending_effects)
        if not _queue_valid(engine, pending, actors):
            return None
        word = "".join("R" if job.racer_id == romantic.racer_id else "C" for job in pending)
        control = copy.position, copy.copied_target == romantic.racer_id, word
        if _third_wheel12(actors):
            if not word or any(token not in "RC" for token in word) or "R" not in word:
                return None
            _renewing_proof(profile)
            profile = (THIRD_WHEEL12_PROFILE, profile)
        elif control not in _finite_proof(profile)[0]:
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger}, sort_keys=True)
        state = (engine.track, engine._state_key(include_pending=False, include_decisions=False), roster,
                 engine.score_ledger.snapshot(), tuple(engine.turns), engine.race_index, tuple(engine.turn_order),
                 tuple(engine.remaining_abilities), tuple(engine.egg_drawn), tuple(engine.previous_winners),
                 engine._preparing, engine._turn_in_progress, getattr(engine, "_resolution_depth", 0),
                 tuple(sorted(getattr(engine, "_duel_offered_epochs", {}).items())),
                 initial, engine._score_start, engine.initial_code_fingerprint, engine.max_events, profile)
        source = _copy_source(engine, copy)
        if source is None:
            return None
        if source:
            # Preserve state[3] (ledger) and the final profile slot.
            state = (*state[:-1], ("static_copy", copy.racer_id, source), state[-1])
        hash(state)
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return state


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple[PendingEffect, ...]
    arrival_counter: int
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        raise ValueError("not a certified Romantic anchor/Copy FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine._arrival_counter, tuple(engine.events), tuple(engine.decisions))


class _Reject(Exception):
    pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot, self.actors = engine, snapshot, actors
        romantic, copy, anchor, spectator = actors
        self.r, self.c, self.p, self.s = (actor.racer_id for actor in actors)
        self.positions = {actor.racer_id: actor.position for actor in actors}
        self.target = copy.copied_target
        self.order = tuple(engine._clockwise_order())
        self.events = engine.events[snapshot.event_start:]
        self.offset, self.decision, self.batch = 0, snapshot.decision_start, snapshot.arrival_counter
        self.children, self.choices = [], []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def arrival(self, actor, destination, steps, cause, *, arrow=False):
        before = self.positions[actor]
        self.batch += 1
        values = dict(before=before, after=destination, amount=steps,
                      movement_kind="track" if arrow else "ability", parent_sequence=cause.sequence,
                      arrival_batch_id=self.batch)
        if not arrow:
            values.update(source_id=actor, ability=Ability.ROMANTIC.value)
        move = self.take("move", actor, **values)
        self.positions[actor] = destination
        stop = self.take("stop", actor, before=before, after=destination,
                         parent_sequence=move.sequence, arrival_batch_id=self.batch)
        if (actor == self.r and destination == 22
                or _third_wheel12(self.actors) and actor == self.c and destination == 12):
            for rid in self.order:
                if rid == self.r or rid == self.c and self.target == self.r:
                    event = self.take("effect_queued", rid, source_id=rid, target_id=rid,
                                      ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
                    self.children.append(PendingEffect(rid, rid, stop.sequence, event.sequence,
                                                       "move", 2, Ability.ROMANTIC.value, rid))
        if destination == 24:
            self.arrival(actor, 22, -2, stop, arrow=True)
        elif destination == 16:
            self.arrival(actor, 12, -4, stop, arrow=True)

    def refresh(self):
        by_id = {actor.racer_id: actor for actor in self.actors}
        anchor = by_id[self.p]
        targets = tuple(rid for rid in self.order if rid in {self.r, self.p})
        legal = (Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets))
        if self.decision >= len(self.engine.decisions):
            raise _Reject
        public = tuple(PublicRacer(
            rid, by_id[rid].name, self.positions[rid], "active",
            (Ability.ROMANTIC if self.target == self.r else anchor.ability).value
            if rid == self.c else by_id[rid].ability.value,
            by_id[rid].tripped, None, self.target if rid == self.c else None)
            for rid in self.engine.turn_order)
        observation = Observation(
            self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            WILD_WILDS.name, WILD_WILDS.finish_position, WILD_WILDS.second_corner_position,
            tuple(sorted(WILD_WILDS.trip_positions)), public, (), None, None,
            score_totals=self.snapshot.state[3].totals)
        request = DecisionRequest(self.decision + 1, self.c, DecisionWindow.COPY_CHOICE, observation, legal)
        record = self.engine.decisions[self.decision]
        if record.request != request or record.action not in legal:
            raise _Reject
        event = self.take("decision", self.c, source_id=record.action.source_id, target_id=record.action.target_id,
                          action_kind=record.action.kind.value, decision_id=request.decision_id)
        if event.sequence != record.event_sequence:
            raise _Reject
        old = self.target
        if record.action.kind != ActionKind.KEEP:
            self.target = record.action.target_id
        if self.target != old:
            self.take("copy_changed", self.c, source_id=self.c, target_id=self.target,
                      ability=Ability.ROMANTIC.value if self.target == self.r else anchor.ability.value)
        self.choices.append(event.sequence)
        self.decision += 1
        return "keep" if record.action.kind == ActionKind.KEEP else "copy_r" if self.target == self.r else "copy_p"

    def job(self, pending):
        before = self.positions[self.c], self.target == self.r
        self.children = []
        actor = pending.racer_id
        power = self.take("ability_effect", actor, source_id=actor, ability=Ability.ROMANTIC.value,
                          amount=2, parent_sequence=pending.queued_sequence)
        self.arrival(actor, self.positions[actor] + 2, 2, power)
        action = self.refresh()
        return tuple(self.children), before, action


def completed(engine, snapshot):
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    # Reauthenticate every original field/provenance, while allowing the real
    # consumption that this completed period must prove. Current pending IDs
    # still pass the stricter unconsumed gate in key().
    try:
        if not _queue_valid(engine, snapshot.original_pending, actors, require_unconsumed=False):
            return False
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    events = engine.events[snapshot.event_start:]
    if not events or any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(events)):
        return False
    reader, fifo = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
    renewing = _third_wheel12(actors)
    profile = snapshot.state[-1][1] if renewing else snapshot.state[-1]
    if renewing:
        edges, table_sha = _renewing_proof(profile)
        states = ()
    else:
        states, edges, table_sha = _finite_proof(profile)
    labels = {actors[0].racer_id: "R", actors[1].racer_id: "C"}
    executed, counts = [], {"R": 0, "C": 0}
    try:
        while reader.offset < len(events):
            word = "".join(labels[job.racer_id] for job in fifo)
            state = reader.positions[reader.c], reader.target == reader.r, word
            if (renewing and (not word or "R" not in word or any(token not in "RC" for token in word))
                    or not renewing and state not in states):
                return False
            pending = fifo.popleft()
            children, before, action = reader.job(pending)
            if before != state[:2]:
                return False
            executed.append(pending.queued_sequence)
            counts[labels[pending.racer_id]] += 1
            fifo.extend(children)
            after = reader.positions[reader.c], reader.target == reader.r, "".join(labels[job.racer_id] for job in fifo)
            expected = _renewing_successor(state, action, profile) if renewing else _successor(state, action, profile)
            if after != expected:
                return False
    except (KeyError, ValueError, IndexError, _Reject):
        return False
    original = [job.queued_sequence for job in snapshot.original_pending]
    fresh = [job.queued_sequence for job in fifo]
    if (reader.decision != len(engine.decisions) or reader.batch != engine._arrival_counter
            or [identity for identity in executed if identity in original] != original
            or len(executed) != len(set(executed)) or set(original).intersection(fresh)
            or any(identity <= snapshot.event_start for identity in fresh)
            or tuple(fifo) != tuple(engine.pending_effects) or not counts["R"]
            or reader.positions != {actor.racer_id: actor.position for actor in actors}
            or reader.target != actors[1].copied_target
            or (sum(job.racer_id == actors[0].racer_id for job in fifo) < 1 if renewing
                else sum(job.racer_id == actors[0].racer_id for job in fifo) != 1)):
        return False
    evidence = {"original_pending_ids": original, "executed_pending_ids": executed, "new_pending_ids": fresh,
            "queue_before": len(original), "queue_after": len(fresh), "job_counts": counts,
            "copy_decision_event_ids": reader.choices, "clockwise_profile": snapshot.state[-1],
            "finite_configurations": len(states), "all_legal_action_edges": len(edges), "finite_table_sha256": table_sha,
            "original_romantic_id": actors[0].racer_id, "copy_id": actors[1].racer_id,
            "anchor_id": actors[2].racer_id, "anchor_ability": actors[2].ability.value,
            "spectator_id": actors[3].racer_id, "spectator_ability": actors[3].ability.value,
            "scores_unchanged": True, "mechanical_state_returned": True, "all_original_jobs_executed": True,
            "all_actual_copy_choices_retained": True, "only_fresh_pending_descendants_stopped": True,
            "persistent_romantic_jobs": 1,
            "proof": "all-choice finite FIFO closed family; persistent original Romantic; real whole cohort and full mechanical/score state authenticated"}
    source = _copy_source(engine, actors[1])
    if renewing:
        evidence.pop("finite_configurations")
        evidence.pop("all_legal_action_edges")
        evidence.pop("finite_table_sha256")
        evidence.update(proof_domain=THIRD_WHEEL12_PROFILE, clockwise_profile=profile,
                        control_configurations=8, all_legal_control_edges=len(edges),
                        production_table_sha256=table_sha, queue_length_unbounded=True,
                        persistent_romantic_jobs=sum(job.racer_id == actors[0].racer_id for job in fifo),
                        proof="all-choice FIFO production grammar; R-token count never decreases; original jobs and all actual choices served before full mechanical/score return; only fresh descendants stop")
    if source:
        evidence["static_copy"] = _copy_metadata(actors[1], source)
    return evidence

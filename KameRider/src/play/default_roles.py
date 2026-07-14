"""Default dynamic role implementations for SoccerSim.

Each role extends :class:`RoleStrategy` and assembles its subtree through
:meth:`build_subtree`. Methods such as ``target``, ``wants_to_kick``, and
``kick_target`` are implementation helpers for role utility nodes, not
base-class contracts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import py_trees

from ..soccer_framework import PlayContext, Pose2D, ReadySlot
from .nodes import AttackSubtreeConfig, MoveToTarget, build_attack_subtree
from .role import RoleStrategy

if TYPE_CHECKING:
    from ..runtime import SoccerKit


# ----------------------------------------------------------------------
# Chaser: approach and kick the ball
# ----------------------------------------------------------------------

# Approach alignment distance behind the ball while chasing, in meters.
_CHASER_APPROACH_OFFSET = 0.4


class ChaserRole(RoleStrategy):
    """Default chaser; kick target is split by ReadySlot into center shot or side clearance."""

    name = "chaser"

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball
        kt = self.kick_target(kit, player_id, context)
        kick_theta = math.atan2(kt.y - ball.y, kt.x - ball.x)
        return kit.motion.approach_target(
            ball,
            kick_theta,
            _CHASER_APPROACH_OFFSET,
        )

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        slot = kit.config.ready_slot_for_player(player_id)
        if slot == ReadySlot.SIDE:
            return kit.targeting.select_clear_or_pass_target(
                player_id,
                context,
                kit.is_player_allowed,
            )
        return kit.targeting.select_kick_target(
            player_id,
            context,
            kit.is_player_allowed,
        )

    def _approach_reason(self, kit: "SoccerKit", player_id: int) -> str:
        slot = kit.config.ready_slot_for_player(player_id)
        return f"{slot.value} approach ball"

    def _kick_reason(
        self,
        kit: "SoccerKit",
        player_id: int,
        target: Pose2D,
    ) -> str:
        slot = kit.config.ready_slot_for_player(player_id)
        default = "side clear" if slot == ReadySlot.SIDE else "center kick"
        return kit.targeting.kick_reason(target, default=default)

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda context: self.target(kit, player_id, context),
                kick_target_fn=lambda context: self.kick_target(
                    kit,
                    player_id,
                    context,
                ),
                reason_fn=lambda: self._approach_reason(kit, player_id),
                kick_reason_fn=lambda target: self._kick_reason(
                    kit,
                    player_id,
                    target,
                ),
            ),
        )


# ----------------------------------------------------------------------
# Supporter: attacking support position
# ----------------------------------------------------------------------


class SupporterRole(RoleStrategy):
    """Attacking support role; uses :meth:`Targeting.support_target` for positioning.

    When the ball is in our dangerous area (defensive mode active), falls back
    to goal-line guard positioning since normal support would be ineffective.
    """

    name = "supporter"

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        result = kit.targeting.support_target(
            player_id,
            context,
            kit.is_player_allowed,
        )
        # When ball is in dangerous area, normal support positioning returns None.
        # Fall back to goal-line guard to maintain defensive shape.
        if result is None:
            return kit.ready_stance.goalkeeper_guard_target(context.known_ball)
        return result

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return MoveToTarget(
            kit,
            player_id,
            lambda context: self.target(kit, player_id, context),
            reason_fn=lambda: "supporter hold",
            hold_vyaw=0.12,
        )


# ----------------------------------------------------------------------
# Defender: shot-blocking support for goalkeeper in dangerous zone
# ----------------------------------------------------------------------


class DefenderRole(RoleStrategy):
    """Defensive blocking role that positions between ball and goal.

    When the ball is in our dangerous goal area, the defender positions
    itself between the ball and the goal to block shots and cover angles.
    When the ball is NOT in the dangerous area, falls back to normal
    supporter positioning for attacking support.
    """

    name = "defender"

    # Approach alignment distance for defender challenges, slightly tighter than chaser.
    _APPROACH_OFFSET = 0.30

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball
        dangerous = kit.targeting.ball_in_own_defensive_area(ball)

        if dangerous:
            # Use defensive blocking positioning
            defense_target = kit.targeting.goalkeeper_support_target(
                player_id, context, kit.is_player_allowed
            )
            if defense_target is not None:
                return defense_target
            # Fallback to normal support if defensive target unavailable
            return kit.ready_stance.goalkeeper_guard_target(ball)

        # Ball is safe - fall back to normal supporter positioning
        return self._fallback_target(kit, player_id, context)

    def _fallback_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        """Normal supporter positioning when ball is not in dangerous area."""
        ball = context.known_ball
        kt = self._fallback_kick_target(kit, player_id, context)
        kick_theta = math.atan2(kt.y - ball.y, kt.x - ball.x)
        return kit.motion.approach_target(
            ball,
            kick_theta,
            _CHASER_APPROACH_OFFSET,
        )

    def _fallback_kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        """Select kick target when falling back to supporter behavior."""
        slot = kit.config.ready_slot_for_player(player_id)
        if slot == ReadySlot.SIDE:
            return kit.targeting.select_clear_or_pass_target(
                player_id,
                context,
                kit.is_player_allowed,
            )
        return kit.targeting.select_kick_target(
            player_id,
            context,
            kit.is_player_allowed,
        )

    def wants_to_kick(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> bool:
        """Only want to kick when ball is dangerous AND we're close enough to challenge."""
        ball = context.known_ball
        if not kit.targeting.ball_in_own_defensive_area(ball):
            return False
        # Check if we're close enough to challenge
        robot = context.teammates.get(player_id)
        if robot is None or robot.pose is None:
            return False
        distance = math.hypot(ball.x - robot.pose.x, ball.y - robot.pose.y)
        return distance < 1.5

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        """When defending, clear the ball away from our goal."""
        ball = context.known_ball
        if kit.targeting.ball_near_sideline(ball):
            return kit.targeting.sideline_recovery_target(ball)
        return Pose2D(
            kit.field.opponent_goal_x(),
            0.0,
            kit.field.attack_theta(),
        )

    def _defend_reason(self, kit: "SoccerKit", player_id: int) -> str:
        return "defender block"

    def _fallback_reason(self, kit: "SoccerKit", player_id: int) -> str:
        return "defender support"

    def _kick_reason(
        self,
        kit: "SoccerKit",
        player_id: int,
        target: Pose2D,
    ) -> str:
        return kit.targeting.kick_reason(target, default="defender clear")

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        # Store fallback reference for closure capture
        self._fallback = SupporterRole()
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda context: self.target(kit, player_id, context),
                kick_target_fn=lambda context: self.kick_target(
                    kit, player_id, context
                ),
                wants_kick_fn=lambda context: self.wants_to_kick(
                    kit, player_id, context
                ),
                reason_fn=lambda: self._defend_reason(kit, player_id),
                kick_reason_fn=lambda target: self._kick_reason(
                    kit, player_id, target
                ),
                hold_vyaw=0.12,
            ),
        )


# ----------------------------------------------------------------------
# Goalkeeper: guard the goal and clear dangerous balls
# ----------------------------------------------------------------------


class GoalkeeperRole(RoleStrategy):
    """Goalkeeper guarding and defensive-area clearance."""

    name = "goalkeeper"

    # Approach alignment distance for goalkeeper challenges, tighter than the chaser, in meters.
    _APPROACH_OFFSET = 0.22

    def target(
        self,
        kit: "SoccerKit",
        context: PlayContext,
    ) -> Pose2D:
        # When the ball is dangerous and kickable, target the approach point behind the ball to enter IsInKickRange.
        # Otherwise return to the goal-line guard target.
        ball = context.known_ball
        if self.wants_to_kick(kit, context):
            kt = self.kick_target(kit, context)
            kick_theta = math.atan2(kt.y - ball.y, kt.x - ball.x)
            return kit.motion.approach_target(
                ball,
                kick_theta,
                self._APPROACH_OFFSET,
            )
        return kit.ready_stance.goalkeeper_guard_target(ball)

    def wants_to_kick(
        self,
        kit: "SoccerKit",
        context: PlayContext,
    ) -> bool:
        return kit.targeting.ball_in_own_defensive_area(context.known_ball)

    def kick_target(
        self,
        kit: "SoccerKit",
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball
        if kit.targeting.ball_near_sideline(ball):
            return kit.targeting.sideline_recovery_target(ball)
        return Pose2D(
            kit.field.opponent_goal_x(),
            0.0,
            kit.field.attack_theta(),
        )

    def _guard_reason(self) -> str:
        return "goalkeeper guard"

    def _kick_reason(
        self,
        kit: "SoccerKit",
        target: Pose2D,
    ) -> str:
        return kit.targeting.kick_reason(target, default="goalkeeper clear")

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda context: self.target(kit, context),
                kick_target_fn=lambda context: self.kick_target(kit, context),
                wants_kick_fn=lambda context: self.wants_to_kick(kit, context),
                reason_fn=self._guard_reason,
                kick_reason_fn=lambda target: self._kick_reason(kit, target),
                hold_vyaw=0.12,
            ),
        )

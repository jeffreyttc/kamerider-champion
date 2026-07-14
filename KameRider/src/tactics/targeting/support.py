"""Support-position targets plus teammate-spacing pushout.

SafetyGuards ensure PLAY support targets only run with fresh ball and
GameController data; support then stays behind the ball by ``support_depth_m``
and ``support_lateral_m`` and pushes away from teammates to avoid stacking.
"""

from __future__ import annotations

import math

from ...soccer_framework import (
    Pose2D,
    SoccerConfig,
    PlayContext,
)
from ..geometry import clamp
from ..geometry import TeamFieldFrame
from .attack import PlayerAllowed


__all__ = ["goalkeeper_support_target", "support_target"]


def goalkeeper_support_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """Compute a defensive blocking target when the ball is in the dangerous goal area.

    Positions the non-goalkeeper robot between the ball and the goal to block shots.
    The defender positions itself along the line from the goal center through the ball,
    closer to the goal than the ball, at a configurable distance from the goal center.

    Key behaviors:
    - If ball is beyond the block line (further from goal), place defender between ball and goal
    - Y position is interpolated based on ball angle for optimal shot coverage
    - Defender never gets closer to the goal than ``defender_min_x_m``
    - Defender never goes further than ``defender_max_x_m`` from the goal
    - Y is clamped to keep the defender inside the field bounds
    """
    ball = context.known_ball

    # Only activate defensive positioning when ball is in our half
    half_margin = config.field_length / 2.0
    if ball.x >= 0.0:
        # Ball is in our attacking half or midfield; fall back to normal support
        return None

    # Block distance from goal center (negative x = toward our goal)
    min_x = config.strategy.defender_min_x_m
    max_x = config.strategy.defender_max_x_m

    # Determine the ideal block x position based on ball position.
    # The defender positions between the ball and the goal to block shots.
    # We want: min_x <= block_x < ball.x (defender closer to goal than ball)
    if ball.x <= min_x:
        # Ball is extremely close to goal - defender stays at minimum x
        block_x = min_x
    elif ball.x < max_x:
        # Ball is in the danger zone - position between ball and goal
        # Place defender roughly halfway between ball and goal, clamped to bounds
        block_x = max(min_x, (ball.x + min_x) / 2.0)
        # Ensure defender is always between ball and goal
        block_x = min(block_x, ball.x - 0.3)
    else:
        # Ball is beyond the defensive line - hold the line at max_x
        block_x = max_x

    # Calculate y position: interpolate between goal center and ball y
    # This gives us a position that tracks the ball's angle relative to the goal
    goal_center_y = 0.0
    # Use a fraction of the ball's y to position the defender
    # Closer balls get more direct blocking (higher factor), farther balls get wider coverage
    ball_dist_from_goal = abs(ball.x)
    if ball_dist_from_goal < 2.0:
        # Very close ball - block directly in front
        y_factor = 0.9
    elif ball_dist_from_goal < 4.0:
        # Medium distance - partial tracking
        y_factor = 0.6
    else:
        # Farther ball - wider coverage with some tracking
        y_factor = 0.35

    target_y = goal_center_y + (ball.y - goal_center_y) * y_factor

    # Clamp y to field bounds with margin
    y_margin = config.strategy.defender_clamp_y_margin_m
    field_half_width = config.field_width / 2.0
    target_y = clamp(
        target_y,
        -field_half_width + 0.45 + y_margin,
        field_half_width - 0.45 - y_margin,
    )

    target_pose = Pose2D(block_x, target_y, field.face_ball_theta(block_x, target_y, ball))
    target_pose = field.clamp_inside_field(target_pose)

    # Apply teammate spacing pushout to avoid clustering with goalkeeper
    return _spaced_defender_target(
        config,
        field,
        player_id,
        context,
        target_pose,
        is_player_allowed,
    )


def support_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """Compute this tick's supporter target Pose2D.

    Stay behind the ball by ``support_depth_m`` and laterally split by player_id parity.
    When the ball is in the opponent's half, the supporter may follow into attack.
    Otherwise the position is clamped to our half to maintain defensive shape.
    Pushout: use :func:`_spaced_support_target` to avoid overlapping other supporters.
    """

    side = 1.0 if player_id % 2 == 0 else -1.0
    lateral = config.strategy.support_lateral_m * side
    ball = context.known_ball
    x = ball.x - config.strategy.support_depth_m
    # Only clamp to our half when the ball is not deep in the opponent's territory.
    # This allows the supporter to join the attack when the ball is on the opponent side.
    half_margin = config.field_length / 2.0
    if ball.x < half_margin * 0.15:
        x = field.own_half_x(x, margin=0.35)
    y = clamp(
        ball.y + lateral,
        -config.field_width / 2.0 + 0.45,
        config.field_width / 2.0 - 0.45,
    )
    target = field.clamp_inside_field(
        Pose2D(x, y, field.face_ball_theta(x, y, ball))
    )
    return _spaced_support_target(
        config,
        field,
        player_id,
        context,
        target,
        is_player_allowed,
    )


# Defender spacing pushout


def _spaced_defender_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    player_id: int,
    context: PlayContext,
    target: Pose2D,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """Push defender target away from the goalkeeper if too close.

    Unlike general supporter spacing, this only considers the goalkeeper
    as an obstacle since the defender's primary concern is not blocking
    the goalkeeper's movement path.
    """
    min_spacing = config.strategy.support_min_spacing_m
    if min_spacing <= 0.0:
        return target

    ball = context.known_ball
    game = context.known_game

    # Only consider the goalkeeper as a spacing obstacle for the defender
    goalkeeper_id = config.goalkeeper_player_id()
    teammate_poses = tuple(
        robot.pose
        for pid, robot in context.teammates.items()
        if pid != player_id and pid != goalkeeper_id
        and robot.pose is not None
        and is_player_allowed(game, pid)
    )

    # Also check goalkeeper distance separately
    keeper_pose = None
    if goalkeeper_id is not None:
        keeper_robot = context.teammates.get(goalkeeper_id)
        if keeper_robot is not None and keeper_robot.pose is not None:
            if is_player_allowed(game, goalkeeper_id):
                keeper_pose = keeper_robot.pose

    # Find closest relevant teammate (non-goalkeeper first)
    closest = None
    if teammate_poses:
        closest = min(
            teammate_poses,
            key=lambda pose: math.hypot(pose.x - target.x, pose.y - target.y),
        )

    # If no non-keeper teammate, use goalkeeper
    if closest is None and keeper_pose is not None:
        closest = keeper_pose
    elif keeper_pose is not None:
        # Use whichever is closer between closest teammate and goalkeeper
        keeper_dist = math.hypot(keeper_pose.x - target.x, keeper_pose.y - target.y)
        current_dist = math.hypot(closest.x - target.x, closest.y - target.y)
        if keeper_dist < current_dist:
            closest = keeper_pose

    if closest is None:
        return target

    dx = target.x - closest.x
    dy = target.y - closest.y
    distance = math.hypot(dx, dy)
    if distance >= min_spacing:
        return target

    if distance <= 1e-6:
        # Target overlaps the closest robot; push laterally
        lane_sign = 1.0 if target.y >= ball.y else -1.0
        if abs(target.y - ball.y) < 1e-6:
            lane_sign = 1.0 if player_id % 2 == 0 else -1.0
        dx, dy = 0.0, lane_sign
        distance = 1.0

    scale = min_spacing / distance
    pushed = field.clamp_inside_field(
        Pose2D(
            closest.x + dx * scale,
            closest.y + dy * scale,
            target.theta,
        )
    )
    return Pose2D(
        pushed.x,
        pushed.y,
        field.face_ball_theta(pushed.x, pushed.y, ball),
    )


# Teammate spacing pushout for regular supporters


def _spaced_support_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    player_id: int,
    context: PlayContext,
    target: Pose2D,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """If target is closer than min_spacing to the nearest teammate, push it along "teammate -> target" out to ``min_spacing``.

    Steps:
    1. Find the nearest legal teammate.
    2. If distance is large enough, do nothing.
    3. Otherwise scale the "teammate -> target" unit vector to min_spacing.
    4. Clamp inside the field and finally face the ball.

    Degenerate case: when target almost overlaps the teammate, no direction can
    be scaled, so fall back to ``lane_sign`` based on which side of the ball target
    is on; if target is exactly on the ball, split by player_id parity.

    In extreme corners with teammate pressure, clamping can make the final target
    slightly closer than min_spacing. With at most three teammates this is rare; if
    strict final distance is needed, iterate once more after clamping.
    """

    min_spacing = config.strategy.support_min_spacing_m
    if min_spacing <= 0.0:
        return target

    ball = context.known_ball
    game = context.known_game
    teammate_poses = tuple(
        robot.pose
        for teammate_id, robot in context.teammates.items()
        if teammate_id != player_id
        and robot.pose is not None
        and is_player_allowed(game, teammate_id)
    )
    if not teammate_poses:
        return target

    closest = min(
        teammate_poses,
        key=lambda pose: math.hypot(pose.x - target.x, pose.y - target.y),
    )
    dx = target.x - closest.x
    dy = target.y - closest.y
    distance = math.hypot(dx, dy)
    if distance >= min_spacing:
        return target

    if distance <= 1e-6:
        # Target overlaps the nearest teammate; use the original lane_sign fallback direction.
        lane_sign = 1.0 if target.y >= ball.y else -1.0
        if abs(target.y - ball.y) < 1e-6:
            lane_sign = 1.0 if player_id % 2 == 0 else -1.0
        dx, dy = 0.0, lane_sign
        distance = 1.0

    scale = min_spacing / distance
    pushed = field.clamp_inside_field(
        Pose2D(
            closest.x + dx * scale,
            closest.y + dy * scale,
            target.theta,
        )
    )
    return Pose2D(
        pushed.x,
        pushed.y,
        field.face_ball_theta(pushed.x, pushed.y, ball),
    )

# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard run-name helpers shared by RSL-RL training entry points."""

from __future__ import annotations

import os
import re
from datetime import datetime


def sanitize_log_component(value: str, fallback: str) -> str:
    """Convert a task or action label into a stable filesystem-friendly component."""
    component = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return component or fallback


def _motion_action_name(env_cfg: object) -> str | None:
    commands = getattr(env_cfg, "commands", None)
    motion = getattr(commands, "motion", None)
    motion_file = getattr(motion, "motion_file", "")
    if motion_file:
        return os.path.splitext(os.path.basename(os.fspath(motion_file)))[0]

    motion_data = getattr(env_cfg, "motion_data", None)
    motion_dataset = getattr(motion_data, "motion_dataset", None)
    motion_data_dir = getattr(motion_dataset, "motion_data_dir", "")
    if motion_data_dir:
        return os.path.basename(os.path.normpath(os.fspath(motion_data_dir)))
    return None


def resolve_action_name(task_name: str, env_cfg: object, override: str | None = None) -> str:
    """Resolve the action component, preferring a CLI override and concrete motion data."""
    if override and override.strip():
        return sanitize_log_component(override, "training")

    motion_action = _motion_action_name(env_cfg)
    if motion_action:
        return sanitize_log_component(motion_action, "motion")

    task_slug = sanitize_log_component(task_name.split(":")[-1], "task")
    action_labels = (
        (("parkour",), "parkour"),
        (("attn", "enc"), "terrain_attention_locomotion"),
        (("interrupt",), "interrupt_locomotion"),
        (("rough",), "rough_locomotion"),
        (("flat",), "flat_locomotion"),
        (("amp",), "amp_motion_set"),
        (("getup",), "getup"),
        (("beyondmimic",), "motion_tracking"),
    )
    for required_tokens, action_label in action_labels:
        if all(token in task_slug for token in required_tokens):
            return action_label
    return "training"


def build_standard_run_name(
    task_name: str,
    action_name: str,
    timestamp: datetime | None = None,
) -> str:
    """Build ``task__action__YYYYMMDD_HHMMSS`` using local wall-clock time."""
    task_component = sanitize_log_component(task_name.split(":")[-1], "task")
    action_component = sanitize_log_component(action_name, "training")
    time_component = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{task_component}__{action_component}__{time_component}"

"""Behavior of the pinned SSv2 evaluation sampler, cache token layouts and explicit root mapping."""
from pathlib import Path

import pytest

from compression.extract import (CAMERA, CLS, PATCH, REGISTER, PathMap, image_layout,
                                 video_eval_clip_indices, view_group_layout)


def test_short_clip_pads_with_last_valid_frame_and_segments_coincide():
    # 30 frames: upstream linspace(0, 29, 7), padded with 29, clipped to 28, truncated.
    expected = [0, 4, 9, 14, 19, 24] + [28] * 10
    first, branch = video_eval_clip_indices(30, segment=0)
    second, _ = video_eval_clip_indices(30, segment=1)
    assert branch == 'overlapping_short_pad'
    assert first.tolist() == expected
    assert second.tolist() == expected


def test_overlapping_segments_shift_by_clip_step_and_stay_in_bounds():
    first, _ = video_eval_clip_indices(100, segment=0)
    second, _ = video_eval_clip_indices(100, segment=1)
    assert first[0] == 0 and first[-1] == 62
    assert (second - first).tolist() == [36] * 16
    assert second.max() <= 99
    # A partition exactly clip_len long is not "longer" upstream; it uses the overlap branch.
    equal, branch = video_eval_clip_indices(128, segment=1)
    assert branch == 'overlapping_short_pad' and equal.max() == 126


def test_long_clip_uses_deterministic_partition_window():
    expected = [0, 4, 8, 12, 17, 21, 25, 29, 34, 38, 42, 46, 51, 55, 59, 63]
    first, branch = video_eval_clip_indices(200, segment=0)
    second, _ = video_eval_clip_indices(200, segment=1)
    assert branch == 'partition_window'
    assert first.tolist() == expected
    assert second.tolist() == [index + 100 for index in expected]


def test_minimum_length_boundary():
    assert video_eval_clip_indices(2)[0].tolist() == [0] * 16
    with pytest.raises(ValueError):
        video_eval_clip_indices(1)


def unique_rows(positions, types):
    return len({(int(t), *map(int, p)) for p, t in zip(positions, types)}) == len(types)


def test_image_layout_is_native_cls_registers_patches_order():
    positions, types = image_layout((24, 24), 4)
    assert types[0] == CLS and positions[0].tolist() == [0, 0, 0]
    assert types[1:5].tolist() == [REGISTER] * 4 and positions[1:5, 2].tolist() == [0, 1, 2, 3]
    patches = positions[types == PATCH]
    assert len(patches) == 576 and patches[1].tolist() == [0, 0, 1] and patches[24].tolist() == [0, 1, 0]
    assert unique_rows(positions, types)


def test_view_group_layout_keeps_per_view_camera_registers_patches():
    positions, types = view_group_layout(2, (3, 4), 16)
    block = 1 + 16 + 12
    assert len(types) == 2 * block
    for view in range(2):
        start = view * block
        assert types[start] == CAMERA and positions[start].tolist() == [view, 0, 0]
        assert types[start + 1:start + 17].tolist() == [REGISTER] * 16
        assert positions[start + 1:start + 17, 2].tolist() == list(range(16))
        assert (positions[start:start + block, 0] == view).all()
        assert positions[start + 17:start + block].tolist() == [[view, r, c] for r in range(3) for c in range(4)]
    assert unique_rows(positions, types)


def test_path_map_uses_longest_prefix_on_component_boundaries():
    mapping = PathMap([('/home/u/mnt/data', '/data/shared/data'),
                       ('/home/u/mnt/data/project', '/scratch/project')])
    assert mapping('/home/u/mnt/data/other/b.jpg') == Path('/data/shared/data/other/b.jpg')
    assert mapping('/home/u/mnt/database/c.jpg') == Path('/home/u/mnt/database/c.jpg')
    assert mapping('/home/u/mnt/data/project/a.jpg') == Path('/scratch/project/a.jpg')

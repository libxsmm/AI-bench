#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2024 abetusk
# Source: https://github.com/jakubcerveny/gilbert/blob/9b080a74c5e3b6fe189785c52742f83ac85ba181/gilbert_d2xy.py
# Ported to cutile Copyright (c) Intel Corporation.


def _sgn(value):
    return -1 if value < 0 else (1 if value > 0 else 0)


def _gilbert_d2xy_recursive(dst_idx, cur_idx, x, y, ax, ay, bx, by):
    width = abs(ax + ay)
    height = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    delta = dst_idx - cur_idx

    if height == 1:
        return x + dax * delta, y + day * delta
    if width == 1:
        return x + dbx * delta, y + dby * delta

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    width2 = abs(ax2 + ay2)
    height2 = abs(bx2 + by2)

    if 2 * width > 3 * height:
        if width2 % 2 and width > 2:
            ax2, ay2 = ax2 + dax, ay2 + day

        next_idx = cur_idx + abs((ax2 + ay2) * (bx + by))
        if cur_idx <= dst_idx < next_idx:
            return _gilbert_d2xy_recursive(
                dst_idx, cur_idx, x, y, ax2, ay2, bx, by
            )
        return _gilbert_d2xy_recursive(
            dst_idx,
            next_idx,
            x + ax2,
            y + ay2,
            ax - ax2,
            ay - ay2,
            bx,
            by,
        )

    if height2 % 2 and height > 2:
        bx2, by2 = bx2 + dbx, by2 + dby

    next_idx = cur_idx + abs((bx2 + by2) * (ax2 + ay2))
    if cur_idx <= dst_idx < next_idx:
        return _gilbert_d2xy_recursive(
            dst_idx, cur_idx, x, y, bx2, by2, ax2, ay2
        )
    cur_idx = next_idx

    next_idx = cur_idx + abs((ax + ay) * ((bx - bx2) + (by - by2)))
    if cur_idx <= dst_idx < next_idx:
        return _gilbert_d2xy_recursive(
            dst_idx,
            cur_idx,
            x + bx2,
            y + by2,
            ax,
            ay,
            bx - bx2,
            by - by2,
        )

    return _gilbert_d2xy_recursive(
        dst_idx,
        next_idx,
        x + (ax - dax) + (bx2 - dbx),
        y + (ay - day) + (by2 - dby),
        -bx2,
        -by2,
        -(ax - ax2),
        -(ay - ay2),
    )


def gilbert_d2xy(index, width, height):
    if width >= height:
        return _gilbert_d2xy_recursive(index, 0, 0, 0, width, 0, 0, height)
    return _gilbert_d2xy_recursive(index, 0, 0, 0, 0, height, width, 0)

"""
内部ヘルパ群（API 非公開）。

どこで: `api.sketch_runner`
何を: `api.sketch` の補助（純粋関数/初期化ヘルパ）を分離し、
      `run_sketch` 本体を薄く保つための内部モジュール群。
なぜ: シンプルさと可読性を維持しつつ、責務を小分割するため。
"""

from __future__ import annotations

__all__: list[str] = []

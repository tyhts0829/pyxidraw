"""
どこで: `engine.export.service`。
何を: G-code エクスポートを非ブロッキングで実行する単一ワーカースレッド＋ジョブ管理。
なぜ: UI の滑らかさを維持しつつ、保存処理（重い I/O/変換）をバックグラウンドに逃がすため。

Stage 2: 疑似ジョブ（モック）での進捗/キャンセル/完了フローを提供する。
Stage 3 以降で `GCodeWriter` 実装に接続する。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from util.paths import ensure_gcode_dir


@dataclass(frozen=True)
class Progress:
    state: Literal[
        "pending",
        "running",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    ]
    done_vertices: int
    total_vertices: int
    path: Optional[Path]
    error: Optional[str]


@dataclass
class _Job:
    job_id: str
    coords: np.ndarray
    offsets: np.ndarray
    # 追加メタ（将来: canvas_mm 等）
    params: object | None
    simulate: bool
    name_prefix: Optional[str] = None
    # 進捗
    done_vertices: int = 0
    total_vertices: int = 0
    state: Literal[
        "pending",
        "running",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    ] = "pending"
    error: Optional[str] = None
    path: Optional[Path] = None
    part_path: Optional[Path] = None
    cancel_event: threading.Event = threading.Event()


class ExportService:
    """G-code エクスポート用の単一ワーカースレッド＋ジョブ管理。"""

    def __init__(self, *, writer: object | None = None) -> None:
        self._q: "queue.Queue[_Job]" = queue.Queue(maxsize=1)  # 同時1件 + ペンディング1件
        self._th = threading.Thread(target=self._worker, name="GCodeExportWorker", daemon=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._writer = writer  # Stage 3: GCodeWriter を受け取り、write を呼ぶ
        self._th.start()

    # --- public API ---
    def submit_gcode_job(
        self,
        snapshot: tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray],
        params: object | None = None,
        *,
        simulate: bool = False,
        name_prefix: Optional[str] = None,
    ) -> str:
        """G-code エクスポートジョブを投入し、`job_id` を返す。

        - 既にキューが満杯（実行中/保留中）なら `RuntimeError`。
        - `simulate=True` の場合、疑似ジョブとしてダミーの進捗を生成して .part を書き出す。
        """
        coords, offsets = _unpack_snapshot(snapshot)
        job_id = _new_job_id()
        job = _Job(
            job_id=job_id,
            coords=np.ascontiguousarray(coords, dtype=np.float32),
            offsets=np.ascontiguousarray(offsets, dtype=np.int32),
            params=params,
            simulate=simulate,
            name_prefix=name_prefix,
        )
        job.total_vertices = int(job.coords.shape[0])
        with self._lock:
            self._jobs[job_id] = job
        try:
            self._q.put_nowait(job)
        except queue.Full as e:  # 即座に満杯→上位は「実行中」を表示
            with self._lock:
                self._jobs.pop(job_id, None)
            raise RuntimeError("G-code エクスポートは実行中です") from e
        return job_id

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        job.state = "cancelling"
        job.cancel_event.set()

    def progress(self, job_id: str) -> Progress:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return Progress("failed", 0, 0, None, "unknown job")
        return Progress(job.state, job.done_vertices, job.total_vertices, job.path, job.error)

    # --- worker loop ---
    def _worker(self) -> None:
        while True:
            job = self._q.get()
            try:
                self._run_job(job)
            finally:
                self._q.task_done()

    def _run_job(self, job: _Job) -> None:
        out_dir = ensure_gcode_dir()
        # 推奨: パラメータからキャンバス寸法を取得
        width_mm: Optional[float] = None
        height_mm: Optional[float] = None
        try:
            # 遅延 import 回避のため isinstance は使わずダックタイピング
            width_mm = getattr(job.params, "canvas_width_mm", None)  # type: ignore[assignment]
            height_mm = getattr(job.params, "canvas_height_mm", None)  # type: ignore[assignment]
        except Exception:
            width_mm = None
            height_mm = None
        final_path = _make_gcode_filename(out_dir, width_mm, height_mm, job.name_prefix)
        part_path = final_path.with_suffix(final_path.suffix + ".part")
        job.part_path = part_path
        job.state = "running"
        # 疑似ジョブ: 進捗を 0→100% に単調増加。キャンセル対応。
        if job.simulate:
            try:
                with part_path.open("w", encoding="utf-8") as fp:
                    fp.write("; Simulated G-code export\n")
                    total = max(1, job.total_vertices)
                    step = max(1, total // 50)
                    for i in range(0, total, step):
                        if job.cancel_event.is_set():
                            job.state = "cancelled"
                            raise _Cancelled()
                        # ダミー出力（I/O チャンク）
                        fp.write(f"; chunk up to vertex {i}\n")
                        fp.flush()
                        job.done_vertices = min(i + step, total)
                        time.sleep(0.01)  # 疑似進捗
                # 完了
                part_path.replace(final_path)
                job.path = final_path
                job.state = "completed"
                return
            except _Cancelled:
                # 後段で cleanup
                pass
            except Exception as e:  # 予期せぬ失敗
                job.state = "failed"
                job.error = str(e)
            finally:
                # キャンセル/失敗時の .part 片付け
                try:
                    if part_path.exists():
                        part_path.unlink()
                except Exception:
                    pass
                with self._lock:
                    # completed 以外は path は None に戻す
                    if job.state != "completed":
                        job.path = None
                return

        # Stage 3: 実 writer 実行（本体は未実装想定）
        try:
            if self._writer is None:
                raise NotImplementedError("GCodeWriter が未設定です")
            # 遅延 import 避け（型は緩く扱う）。ファイルは .part に書き出して最後に rename。
            with part_path.open("w", encoding="utf-8") as fp:  # type: ignore[call-arg]
                # 実 writer 呼び出し（本体未実装→例外の想定）。
                # 進捗はここでは終端のみ（骨組み）。
                self._writer.write(job.coords, job.offsets, job.params, fp)  # type: ignore[attr-defined]
                job.done_vertices = job.total_vertices
            part_path.replace(final_path)
            job.path = final_path
            job.state = "completed"
            return
        except NotImplementedError as e:
            job.state = "failed"
            job.error = str(e)
            try:
                if part_path.exists():
                    part_path.unlink()
            except Exception:
                pass
        except Exception as e:
            job.state = "failed"
            job.error = str(e)
            try:
                if part_path.exists():
                    part_path.unlink()
            except Exception:
                pass
        finally:
            with self._lock:
                if job.state != "completed":
                    job.path = None


def _unpack_snapshot(
    snapshot: tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(snapshot, tuple):
        return snapshot
    if isinstance(snapshot, dict):
        return snapshot["coords"], snapshot["offsets"]
    raise TypeError("snapshot は (coords, offsets) か {'coords','offsets'} のみ対応")


def _new_job_id() -> str:
    # タイムスタンプ + スレッドID（簡素で十分）
    return f"job_{int(time.time()*1000)}_{threading.get_ident()}"


def _make_gcode_filename(
    out_dir: Path,
    width_mm: Optional[float],
    height_mm: Optional[float],
    name_prefix: Optional[str] = None,
) -> Path:
    # prefix の有無でタイムスタンプ形式を切替
    ts_prefixed = datetime.now().strftime("%y%m%d_%H%M%S")
    ts_fallback = datetime.now().strftime("%Y%m%d_%H%M%S")

    def dims_tag() -> str:
        if width_mm and width_mm > 0 and height_mm and height_mm > 0:
            return f"{int(round(width_mm))}x{int(round(height_mm))}"
        return "unknownWxH"

    if name_prefix:
        base = f"{name_prefix}_{dims_tag()}_{ts_prefixed}"
        path = out_dir / f"{base}.gcode"
        return _unique_path(path)
    # 旧仕様に近いフォールバック（_mm サフィックスを維持）
    base = f"{ts_fallback}_{dims_tag()}_mm"
    path = out_dir / f"{base}.gcode"
    return _unique_path(path)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}-{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


class _Cancelled(Exception):
    pass

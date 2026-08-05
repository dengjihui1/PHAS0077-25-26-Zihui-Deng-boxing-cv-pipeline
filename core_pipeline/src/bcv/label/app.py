"""FastAPI app for the labelling GUI. One server instance == one video/session.

Routes: the canvas page (``/``), frame JPEGs (``/frame/{idx}``), session meta, and the
bbox-placer label API (get/set/delete keyframes, save+export, model pre-fill). The session
holds the open video, the editable keyframe project, and the output paths.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from ..common.rounds import save_rounds
from ..stage2_crop.cropper import Stage2Config, crop_box_for_frame
from .boxes import BBoxProject, export_fighter_bboxes, save_project
from .frames import FrameSource

STATIC = Path(__file__).parent / "static"


@dataclass
class PrefillState:
    running: bool = False
    done: int = 0
    total: int = 0
    added: int = 0
    error: str | None = None
    _cancel: bool = False
    _thread: threading.Thread | None = None


@dataclass
class Session:
    bout: int
    split: int
    mode: str
    frames: FrameSource
    project: BBoxProject
    project_file: Path
    export_file: Path
    rounds_file: Path         # FIGHT-level rounds.json (shared across the bout's splits)
    pipeline: Any = None      # PipelineConfig (for pre-fill); optional
    stage_cfg: Any = None     # Stage1Config (for pre-fill); optional
    prefill: PrefillState = field(default_factory=PrefillState)
    rounds: list = field(default_factory=list)   # [(start,end), ...]
    pending_start: int | None = None             # an opened-but-not-closed round mark
    candidate_detector: Any = None               # lazy raw-YOLO detector (person boxes)
    crop_cfg: Stage2Config = field(default_factory=Stage2Config)  # for the crop-box preview


class BoxEdit(BaseModel):
    fighter: str
    frame: int
    box: list[int] | None = None  # None = explicit ABSENT keyframe


class KeyRef(BaseModel):
    fighter: str
    frame: int


class RoundMark(BaseModel):
    edge: str   # 'start' | 'end'
    frame: int


class RoundRef(BaseModel):
    index: int


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="bcv-label")
    app.state.session = session

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text())

    @app.get("/api/meta")
    def meta() -> dict:
        s = session
        return {
            "bout": s.bout, "split": s.split, "mode": s.mode,
            "num_frames": s.frames.num_frames, "fps": s.frames.fps,
            "width": s.frames.width, "height": s.frames.height,
            "can_prefill": s.pipeline is not None and s.stage_cfg is not None,
            "can_candidates": s.stage_cfg is not None,
            "export_file": str(s.export_file),
        }

    @app.get("/frame/{idx}")
    def frame(idx: int) -> Response:
        return Response(content=session.frames.jpeg(idx), media_type="image/jpeg")

    @app.get("/api/boxes/{idx}")
    def boxes(idx: int) -> dict:
        st = session.project.frame_state(idx)
        st["crop_box"] = crop_box_for_frame(
            st["red"]["box"], st["blue"]["box"],
            session.frames.width, session.frames.height, session.crop_cfg,
        )
        st["n_fighters"] = int(st["red"]["box"] is not None) + int(st["blue"]["box"] is not None)
        return st

    @app.get("/api/keyframes")
    def keyframes() -> dict:
        return {c: sorted(int(f) for f in kf) for c, kf in session.project.keyframes.items()}

    @app.get("/api/candidates/{idx}")
    def candidates(idx: int) -> dict:
        """Raw (unfiltered) YOLO person boxes for one frame — click to assign red/blue."""
        if session.stage_cfg is None:
            return {"boxes": [], "available": False}
        if session.candidate_detector is None:
            from .candidates import CandidateDetector
            c = session.stage_cfg
            session.candidate_detector = CandidateDetector(
                c.detector_weights, imgsz=c.det_imgsz, device=c.device, half=c.half)
        frame = session.frames.read(idx)
        return {"boxes": session.candidate_detector.detect(frame), "available": True}

    @app.post("/api/box")
    def set_box(edit: BoxEdit) -> dict:
        if edit.fighter not in ("red", "blue"):
            raise HTTPException(400, "fighter must be 'red' or 'blue'")
        session.project.set_key(edit.fighter, edit.frame, edit.box)
        return session.project.frame_state(edit.frame)

    @app.post("/api/box/delete")
    def del_box(ref: KeyRef) -> dict:
        session.project.del_key(ref.fighter, ref.frame)
        return session.project.frame_state(ref.frame)

    # ---- fight rounds (shared across the bout's splits; used to exclude rest frames) ----
    @app.get("/api/rounds")
    def get_rounds() -> dict:
        return {"rounds": session.rounds, "pending_start": session.pending_start}

    def _persist_rounds() -> None:
        save_rounds(session.rounds_file.parent, session.rounds,
                    fps=session.frames.fps, source="manual")

    @app.post("/api/rounds/mark")
    def mark_round(m: RoundMark) -> dict:
        if m.edge == "start":
            session.pending_start = m.frame
        elif m.edge == "end":
            if session.pending_start is None:
                raise HTTPException(400, "mark a round START first")
            if m.frame <= session.pending_start:
                raise HTTPException(400, "round end must be after its start")
            session.rounds.append((session.pending_start, m.frame))
            session.rounds.sort()
            session.pending_start = None
            _persist_rounds()
        else:
            raise HTTPException(400, "edge must be 'start' or 'end'")
        return {"rounds": session.rounds, "pending_start": session.pending_start}

    @app.post("/api/rounds/delete")
    def delete_round(ref: RoundRef) -> dict:
        if 0 <= ref.index < len(session.rounds):
            session.rounds.pop(ref.index)
            _persist_rounds()
        return {"rounds": session.rounds, "pending_start": session.pending_start}

    @app.post("/api/rounds/clear")
    def clear_rounds() -> dict:
        session.rounds, session.pending_start = [], None
        _persist_rounds()
        return {"rounds": session.rounds, "pending_start": None}

    @app.post("/api/save")
    def save() -> dict:
        save_project(session.project_file, session.project)
        export_fighter_bboxes(session.export_file, session.project)
        return {"saved": str(session.project_file), "exported": str(session.export_file)}

    @app.post("/api/prefill")
    def prefill(stride: int = 15, start_frame: int = 0, count: int | None = None) -> dict:
        """Start a BACKGROUND chain pre-fill over [start_frame, start_frame+count).

        Returns immediately; poll /api/prefill/status, cancel via /api/prefill/cancel.
        ``count=None`` => whole video from start_frame.
        """
        if session.pipeline is None or session.stage_cfg is None:
            raise HTTPException(400, "pre-fill unavailable (need --extra detect + a stage1 config)")
        ps = session.prefill
        if ps.running:
            raise HTTPException(409, "pre-fill already running")
        ps.running, ps.done, ps.added, ps.error, ps._cancel = True, 0, 0, None, False
        ps.total = (count if count is not None else session.frames.num_frames - start_frame)

        def _run() -> None:
            from .prefill import prefill_from_chain
            try:
                def progress(d: int, t: int) -> None:
                    ps.done, ps.total = d, t
                ps.added = prefill_from_chain(
                    session.project, session.pipeline, session.stage_cfg,
                    stride=stride, start_frame=start_frame, max_frames=count,
                    on_progress=progress, should_cancel=lambda: ps._cancel,
                )
                save_project(session.project_file, session.project)
            except Exception as e:  # surface to the UI instead of dying silently
                ps.error = f"{type(e).__name__}: {e}"
            finally:
                ps.running = False

        ps._thread = threading.Thread(target=_run, daemon=True)
        ps._thread.start()
        return {"started": True, "total": ps.total}

    @app.get("/api/prefill/status")
    def prefill_status() -> dict:
        ps = session.prefill
        return {"running": ps.running, "done": ps.done, "total": ps.total,
                "added": ps.added, "error": ps.error}

    @app.post("/api/prefill/cancel")
    def prefill_cancel() -> dict:
        session.prefill._cancel = True
        return {"cancelling": True}

    @app.get("/static/{name}")
    def static_file(name: str) -> FileResponse:
        p = STATIC / name
        if not p.is_file():
            raise HTTPException(404, name)
        return FileResponse(p)

    return app

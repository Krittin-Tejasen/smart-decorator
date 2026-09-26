"""Tests for the progress-reporting pipeline and the /generate-room/jobs endpoints.

Run from backend/:  python -m unittest discover -s tests -v

Every AI call (prompt composer, image providers, critic, segmentation) is
patched out, so these never touch Gemini/Replicate and cost nothing.
"""

import asyncio
import base64
import sys
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import generation


def png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48), (120, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


def png_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes()).decode()


COMPOSE = patch.object(generation, "compose_design_prompt", new=AsyncMock(return_value="A calm room."))

_no_db_write = patch.object(generation, "save_generated_design", new=lambda **_kwargs: None)


def setUpModule():
    # run_generate_room persists every result to Supabase (backend/db.py). If a
    # developer's shell happens to have SUPABASE_* set, these tests must still
    # never write rows or upload images.
    _no_db_write.start()


def tearDownModule():
    _no_db_write.stop()


class PipelineStageTests(unittest.IsolatedAsyncioTestCase):
    async def run_pipeline(self, **kwargs):
        seen: list[tuple[str, str, float]] = []
        args = dict(
            room_type="bedroom", style="japandi", color="warm_oat_cream",
            image_bytes=png_bytes(), mime_type="image/png",
        )
        args.update(kwargs)
        result = await generation.run_generate_room(
            **args, report=lambda stage, message, fraction: seen.append((stage, message, fraction))
        )
        return result, seen

    async def test_mock_provider_skips_the_critic(self):
        with COMPOSE, patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())):
            result, seen = await self.run_pipeline(provider="mock")

        self.assertEqual([s for s, _, _ in seen], ["compose", "generate", "match"])
        self.assertTrue(result["generated_image"].startswith("data:image/png"))
        self.assertNotIn("furniture_segments", result)

    async def test_critic_rejection_is_reported_as_a_retry(self):
        critic = AsyncMock(side_effect=[(False, "door is covered"), (True, "ok")])
        with COMPOSE, \
                patch.object(generation, "generate_with_gemini", new=AsyncMock(return_value=png_data_url())), \
                patch.object(generation, "critique_generated_image", new=critic):
            _, seen = await self.run_pipeline(provider="gemini")

        self.assertEqual(
            [s for s, _, _ in seen],
            ["compose", "generate", "critic", "retry", "critic", "match"],
        )
        retry_message = next(m for s, m, _ in seen if s == "retry")
        self.assertEqual(retry_message, "Refining the design (attempt 2)")

        fractions = [f for _, _, f in seen]
        self.assertEqual(fractions, sorted(fractions), "progress must never go backwards")
        self.assertLess(max(fractions), 1.0, "1.0 is reserved for the finished job")

    async def test_segment_stage_only_appears_when_requested(self):
        fake_seg = SimpleNamespace(items=[], model_dump=lambda: {"items": [], "total": 0})
        with COMPOSE, \
                patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())), \
                patch.object(generation, "run_segmentation", new=AsyncMock(return_value=fake_seg)):
            result, seen = await self.run_pipeline(provider="mock", segment=True)

        self.assertEqual([s for s, _, _ in seen][-2:], ["match", "segment"])
        self.assertEqual(result["furniture_segments"], {"items": [], "total": 0})

    async def test_unreadable_image_fails_before_any_stage(self):
        seen = []
        with self.assertRaises(HTTPException) as ctx:
            await generation.run_generate_room(
                "bedroom", "japandi", "warm_oat_cream", b"definitely not an image", "image/png",
                report=lambda *a: seen.append(a),
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(seen, [])

    async def test_no_report_callback_is_fine(self):
        with COMPOSE, patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())):
            result = await generation.run_generate_room(
                "bedroom", "japandi", "warm_oat_cream", png_bytes(), "image/png", provider="mock"
            )
        self.assertIn("generated_image", result)


class JobEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(generation.router)
        # `with` keeps ONE event loop alive for the whole class; without it each
        # request gets its own loop and the background job task would be killed
        # as soon as the POST returned.
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def setUp(self):
        generation._jobs.clear()

    def start(self, image: bytes | None = None, provider: str = "mock"):
        return self.client.post(
            "/generate-room/jobs",
            data={"room_type": "bedroom", "style": "japandi", "color": "warm_oat_cream", "provider": provider},
            files={"image": ("room.png", image if image is not None else png_bytes(), "image/png")},
        )

    def wait_for(self, job_id: str, statuses: set[str], timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = self.client.get(f"/generate-room/jobs/{job_id}").json()
            if body["status"] in statuses:
                return body
            time.sleep(0.02)
        self.fail(f"job never reached {statuses}; last: {body}")

    def test_job_runs_to_done_and_hands_back_the_result(self):
        with COMPOSE, patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())):
            resp = self.start()
            self.assertEqual(resp.status_code, 202)
            body = self.wait_for(resp.json()["job_id"], {"done", "error"})

        self.assertEqual(body["status"], "done")
        self.assertEqual(body["stage"], "done")
        self.assertEqual(body["progress"], 1.0)
        self.assertTrue(body["result"]["generated_image"].startswith("data:image/png"))
        self.assertEqual(len(body["result"]["products"]), 2)

    def test_progress_is_observable_while_the_job_is_still_running(self):
        async def slow_generate(*_args, **_kwargs):
            await asyncio.sleep(0.6)
            return png_data_url()

        with COMPOSE, patch.object(generation, "generate_with_mock", new=slow_generate):
            job_id = self.start().json()["job_id"]
            running = self.wait_for(job_id, {"running"})
            self.assertEqual(running["status"], "running")
            self.assertIn(running["stage"], {"compose", "generate"})
            self.assertNotIn("result", running, "the big result is only sent once the job is done")
            self.assertEqual(self.wait_for(job_id, {"done"})["status"], "done")

    def test_a_pipeline_failure_becomes_an_error_state_with_its_message(self):
        boom = AsyncMock(side_effect=HTTPException(status_code=502, detail="Gemini did not return an image."))
        with COMPOSE, patch.object(generation, "generate_with_mock", new=boom):
            body = self.wait_for(self.start().json()["job_id"], {"error", "done"})

        self.assertEqual(body["status"], "error")
        self.assertEqual(body["error"], "Gemini did not return an image.")
        self.assertNotIn("result", body)

    def test_cancel_stops_the_running_task(self):
        async def very_slow(*_args, **_kwargs):
            await asyncio.sleep(30)
            return png_data_url()

        with COMPOSE, patch.object(generation, "generate_with_mock", new=very_slow):
            job_id = self.start().json()["job_id"]
            self.wait_for(job_id, {"running"})

            self.assertEqual(self.client.delete(f"/generate-room/jobs/{job_id}").json(), {"status": "cancelled"})
            self.assertEqual(self.client.get(f"/generate-room/jobs/{job_id}").json()["status"], "cancelled")

            task = generation._jobs[job_id].task
            deadline = time.time() + 3
            while not task.done() and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue(task.cancelled(), "the pipeline task itself must be cancelled, not just marked")

    def test_cancelling_a_finished_job_keeps_its_status(self):
        with COMPOSE, patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())):
            job_id = self.start().json()["job_id"]
            self.wait_for(job_id, {"done"})
        self.assertEqual(self.client.delete(f"/generate-room/jobs/{job_id}").json(), {"status": "done"})

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/generate-room/jobs/nope").status_code, 404)
        self.assertEqual(self.client.delete("/generate-room/jobs/nope").status_code, 404)

    def test_unreadable_upload_is_rejected_immediately_and_creates_no_job(self):
        resp = self.start(image=b"not an image")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(generation._jobs, {})

    def test_the_original_blocking_endpoint_still_works(self):
        with COMPOSE, patch.object(generation, "generate_with_mock", new=AsyncMock(return_value=png_data_url())):
            resp = self.client.post(
                "/generate-room",
                data={"room_type": "bedroom", "style": "japandi", "color": "warm_oat_cream", "provider": "mock"},
                files={"image": ("room.png", png_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(sorted(resp.json()), ["generated_image", "products"])


class PurgeTests(unittest.TestCase):
    def setUp(self):
        generation._jobs.clear()

    def test_old_finished_jobs_are_dropped_but_running_ones_are_kept(self):
        now = 10_000.0
        generation._jobs.update(
            {
                "old": generation._Job(id="old", created_at=0, status="done", finished_at=now - generation._JOB_TTL_SECONDS - 1),
                "fresh": generation._Job(id="fresh", created_at=0, status="done", finished_at=now - 5),
                "running": generation._Job(id="running", created_at=0),
            }
        )
        generation._purge_old_jobs(now)
        self.assertEqual(sorted(generation._jobs), ["fresh", "running"])

    def test_the_oldest_finished_jobs_go_first_when_over_the_cap(self):
        now = 10_000.0
        for i in range(generation._MAX_JOBS):
            generation._jobs[f"j{i}"] = generation._Job(id=f"j{i}", created_at=0, status="done", finished_at=now - 100 + i)
        generation._purge_old_jobs(now)
        self.assertLess(len(generation._jobs), generation._MAX_JOBS)
        self.assertNotIn("j0", generation._jobs)
        self.assertIn(f"j{generation._MAX_JOBS - 1}", generation._jobs)


if __name__ == "__main__":
    unittest.main()

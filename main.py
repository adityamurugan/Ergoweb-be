import io
import os
import tempfile
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pose import extract_pose_from_image, extract_pose_from_video, get_annotated_image_from_file
from angles import compute_frame_angles
from rula import compute_rula_score


app = FastAPI(title="ErgoWeb RULA API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
	return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
	if not file:
		raise HTTPException(status_code=400, detail="No file uploaded")

	content_type = (file.content_type or "").lower()
	is_video = content_type.startswith("video/") or (
		(not content_type.startswith("image/")) and file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
	)

	# Persist to a temp file for OpenCV/MediaPipe compatibility
	with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
		binary = await file.read()
		tmp.write(binary)
		tmp_path = tmp.name

	try:
		if is_video:
			landmark_frames = extract_pose_from_video(tmp_path)
		else:
			landmarks = extract_pose_from_image(tmp_path)
			landmark_frames = [landmarks] if landmarks is not None else []

		if not landmark_frames:
			raise HTTPException(status_code=422, detail="No pose detected")

		# Compute angles per frame and aggregate
		frame_angles: List[Dict[str, float]] = [compute_frame_angles(lm) for lm in landmark_frames]
		# Aggregate by mean for stability
		angle_keys = frame_angles[0].keys()
		avg_angles: Dict[str, float] = {
			k: float(sum(f[k] for f in frame_angles) / len(frame_angles)) for k in angle_keys
		}

		score, score_details = compute_rula_score(avg_angles)

		# Annotated preview image
		annotated = get_annotated_image_from_file(tmp_path, is_video=is_video)
		annotated_b64 = None
		if annotated is not None:
			import base64
			import cv2
			ok, buf = cv2.imencode('.jpg', annotated)
			if ok:
				annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

		return JSONResponse(
			content={
				"score": score,
				"details": {
					"angles": avg_angles,
					"rula": score_details,
					"framesAnalyzed": len(frame_angles),
					"fileType": "video" if is_video else "image",
					"annotatedImageJpgBase64": annotated_b64
					}
				}
			)
	finally:
		try:
			os.remove(tmp_path)
		except Exception:
			pass



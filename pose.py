from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


PoseLandmark = mp.solutions.pose.PoseLandmark
_mp_drawing = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


def _landmarks_to_np(landmarks) -> Optional[np.ndarray]:
	if not landmarks:
		return None
	coords = []
	for lm in landmarks:
		coords.append([lm.x, lm.y, lm.z, lm.visibility])
	return np.array(coords, dtype=np.float32)


def extract_pose_from_image(path: str) -> Optional[np.ndarray]:
	img = cv2.imread(path)
	if img is None:
		return None
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
		res = pose.process(img_rgb)
		if not res.pose_landmarks:
			return None
		return _landmarks_to_np(res.pose_landmarks.landmark)


def extract_pose_from_video(path: str, max_frames: int = 300) -> List[np.ndarray]:
	cap = cv2.VideoCapture(path)
	frames: List[np.ndarray] = []
	with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
		count = 0
		while cap.isOpened() and count < max_frames:
			ret, frame = cap.read()
			if not ret:
				break
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			res = pose.process(frame_rgb)
			if res.pose_landmarks:
				lm_np = _landmarks_to_np(res.pose_landmarks.landmark)
				if lm_np is not None:
					frames.append(lm_np)
			count += 1
	cap.release()
	return frames


def get_annotated_image_from_file(path: str, is_video: bool, max_frames: int = 300) -> Optional[np.ndarray]:
	"""Return a BGR image annotated with skeleton from the first frame where pose is detected."""
	if not is_video:
		img = cv2.imread(path)
		if img is None:
			return None
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
			res = pose.process(img_rgb)
			if not res.pose_landmarks:
				return None
			_mp_drawing.draw_landmarks(
				image=img,
				landmark_list=res.pose_landmarks,
				connections=mp.solutions.pose.POSE_CONNECTIONS,
				landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style()
			)
			return img

	# video
	cap = cv2.VideoCapture(path)
	with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
		count = 0
		while cap.isOpened() and count < max_frames:
			ret, frame = cap.read()
			if not ret:
				break
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			res = pose.process(frame_rgb)
			if res.pose_landmarks:
				_mp_drawing.draw_landmarks(
					image=frame,
					landmark_list=res.pose_landmarks,
					connections=mp.solutions.pose.POSE_CONNECTIONS,
					landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style()
				)
				cap.release()
				return frame
			count += 1
	cap.release()
	return None



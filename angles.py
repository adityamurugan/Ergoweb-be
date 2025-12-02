from typing import Dict

import numpy as np


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
	"""Return angle in degrees between vectors v1 and v2."""
	v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
	v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
	cosang = float(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	return float(np.degrees(np.arccos(cosang)))


def _get_point(lm: np.ndarray, idx: int) -> np.ndarray:
	return lm[idx, :3]


def compute_frame_angles(landmarks: np.ndarray) -> Dict[str, float]:
	"""
	Compute a minimal set of joint angles needed for a simplified RULA.
	Expects `landmarks` shape [33, 4] per MediaPipe Pose indexing.
	"""
	# MediaPipe indices
	NOSE = 0
	LEFT_SHOULDER = 11
	RIGHT_SHOULDER = 12
	LEFT_ELBOW = 13
	RIGHT_ELBOW = 14
	LEFT_WRIST = 15
	RIGHT_WRIST = 16
	LEFT_HIP = 23
	RIGHT_HIP = 24
	LEFT_KNEE = 25
	RIGHT_KNEE = 26

	# Use right side by default; if visibility poor, left side could be considered later
	shoulder = _get_point(landmarks, RIGHT_SHOULDER)
	elbow = _get_point(landmarks, RIGHT_ELBOW)
	wrist = _get_point(landmarks, RIGHT_WRIST)
	hip = _get_point(landmarks, RIGHT_HIP)
	knee = _get_point(landmarks, RIGHT_KNEE)
	nose = _get_point(landmarks, NOSE)

	# Upper arm (shoulder flexion/abduction proxy): vector shoulder->elbow vs vertical
	upper_arm_vec = elbow - shoulder
	vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
	shoulder_flex = _angle_between(upper_arm_vec, vertical)

	# Lower arm (elbow flexion): vector elbow->wrist vs vector elbow->shoulder
	forearm_vec = wrist - elbow
	upper_arm_rev = shoulder - elbow
	elbow_flex = _angle_between(forearm_vec, upper_arm_rev)

	# Wrist (neutral ~ straight with forearm): angle between hand (wrist->index approximated by wrist vector to elbow) and forearm
	wrist_neutral = _angle_between(elbow - wrist, forearm_vec)

	# Neck: vector nose->shoulder midpoint vs vertical
	shoulder_mid = ( _get_point(landmarks, LEFT_SHOULDER) + _get_point(landmarks, RIGHT_SHOULDER) ) / 2.0
	neck_vec = nose - shoulder_mid
	neck_flex = _angle_between(neck_vec, vertical)

	# Trunk: hip->shoulder midpoint vs vertical
	hip_mid = ( _get_point(landmarks, LEFT_HIP) + _get_point(landmarks, RIGHT_HIP) ) / 2.0
	trunk_vec = shoulder_mid - hip_mid
	trunk_flex = _angle_between(trunk_vec, vertical)

	return {
		"shoulderFlexionDeg": float(shoulder_flex),
		"elbowFlexionDeg": float(elbow_flex),
		"wristNeutralDeg": float(wrist_neutral),
		"neckFlexionDeg": float(neck_flex),
		"trunkFlexionDeg": float(trunk_flex),
	}



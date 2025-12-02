from typing import Dict, Tuple


def _score_upper_arm(shoulder_flexion_deg: float) -> int:
	if shoulder_flexion_deg < 20:
		return 1
	elif shoulder_flexion_deg < 45:
		return 2
	elif shoulder_flexion_deg < 90:
		return 3
	else:
		return 4


def _score_lower_arm(elbow_flexion_deg: float) -> int:
	if 60 <= elbow_flexion_deg <= 100:
		return 1
	elif 0 <= elbow_flexion_deg < 60 or 100 < elbow_flexion_deg <= 120:
		return 2
	else:
		return 3


def _score_wrist(wrist_neutral_deg: float) -> int:
	if wrist_neutral_deg < 15:
		return 1
	elif wrist_neutral_deg < 35:
		return 2
	else:
		return 3


def _score_neck(neck_flexion_deg: float) -> int:
	if neck_flexion_deg < 10:
		return 1
	elif neck_flexion_deg < 20:
		return 2
	elif neck_flexion_deg < 45:
		return 3
	else:
		return 4


def _score_trunk(trunk_flexion_deg: float) -> int:
	if trunk_flexion_deg < 5:
		return 1
	elif trunk_flexion_deg < 20:
		return 2
	elif trunk_flexion_deg < 60:
		return 3
	else:
		return 4


def compute_rula_score(angles: Dict[str, float]) -> Tuple[int, Dict[str, int]]:
	"""
	Simplified RULA score from mean angles. This is an approximation for demo purposes.
	Returns total score and breakdown per region.
	"""
	upper_arm = _score_upper_arm(angles.get("shoulderFlexionDeg", 0.0))
	lower_arm = _score_lower_arm(angles.get("elbowFlexionDeg", 0.0))
	wrist = _score_wrist(angles.get("wristNeutralDeg", 0.0))
	neck = _score_neck(angles.get("neckFlexionDeg", 0.0))
	trunk = _score_trunk(angles.get("trunkFlexionDeg", 0.0))

	group_a = max(upper_arm, lower_arm) + (1 if wrist >= 2 else 0)
	group_b = max(neck, trunk)
	# Simplified combination
	total = min(7, group_a + group_b)

	return total, {
		"upperArm": upper_arm,
		"lowerArm": lower_arm,
		"wrist": wrist,
		"neck": neck,
		"trunk": trunk,
		"groupA": group_a,
		"groupB": group_b,
	}




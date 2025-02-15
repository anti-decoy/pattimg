import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import math
from pathlib import Path
import signal
import subprocess
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

sigterm = False


def termsighandler(signal, frame):
	global sigterm
	sigterm = True


signal.signal(signal.SIGINT, termsighandler)
signal.signal(signal.SIGQUIT, termsighandler)
signal.signal(signal.SIGTERM, termsighandler)
signal.signal(signal.SIGHUP, termsighandler)


def get_script_path():
	return Path(os.path.dirname(os.path.realpath(__file__)))


def now_time_str():
	from datetime import datetime
	now = datetime.now()
	date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
	return date_time


def in_time(t_start_hour, t_start_min, t_end_hour, t_end_min):
	from datetime import datetime
	now = datetime.now()
	start = datetime(now.year, now.month, now.day, t_start_hour, t_start_min, 0)
	end = datetime(now.year, now.month, now.day, t_end_hour, t_end_min, 0)
	return (now >= start) and (now <= end)


def draw_text_on_image(rgb_image, text):
	# Visualization parameters
	row_size = 50  # pixels
	left_margin = 24  # pixels
	text_color = (0, 0, 0)  # black
	font_size = 0.5
	font_thickness = 1

	text_location = (left_margin, row_size)
	cv2.putText(rgb_image, text, text_location, cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness, cv2.LINE_AA)


def draw_landmarks_on_image(mp_image, detection_result):
	rgb_image = np.copy(mp_image.numpy_view())
	pose_landmarks_list = detection_result.pose_landmarks

	# Loop through the detected poses to visualize.
	for idx in range(len(pose_landmarks_list)):
		pose_landmarks = pose_landmarks_list[idx]

		# Draw the pose landmarks.
		pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
		])

		# print(annotated_image.shape)

		solutions.drawing_utils.draw_landmarks(
			rgb_image,
			pose_landmarks_proto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style())
	return rgb_image


def calc_degree(pose_a, pose_b):
	a = math.fabs(pose_b.y - pose_a.y)
	b = math.fabs(pose_b.x - pose_a.x)
	c = math.sqrt(math.fabs(a * a + b * b))
	# 计算余弦值
	cos_a = (b * b + c * c - a * a) / (2 * b * c)
	# 使用反余弦函数计算角度（结果为弧度）
	angle_a_rad = math.acos(cos_a)
	# 将弧度转换为度
	angle_a_deg = math.degrees(angle_a_rad)
	return angle_a_deg


def isok_pose(pose_landmarks):
	if len(pose_landmarks) < 16:
		print(now_time_str(), "# 判断身体部位是否合理：关键部位不全")
		return False, ""
	# right
	shoulder = pose_landmarks[12]
	elbow = pose_landmarks[14]
	# wrist = pose_landmarks[16]

	"""
	# 肩膀高于肘和手腕
	if (shoulder.y > elbow.y) != (shoulder.y > wrist.y):
		print(now_time_str(), "# 判断身体部位是否合理：右侧高度不合理：shoulder.y=%f elbow.y=%f wrist.y=%f" % (shoulder.y, elbow.y, wrist.y))
		return False, "Right shoulder pose is not high than elbow and wrist: shoulder.y=%f elbow.y=%f wrist.y=%f" % (shoulder.y, elbow.y, wrist.y)
	"""
	# 肩膀高于肘和手腕
	if (shoulder.y > elbow.y):
		print(now_time_str(), "# 判断身体部位是否合理：右侧高度不合理：shoulder.y=%f elbow.y=%f " % (shoulder.y, elbow.y))
		return False, "Right shoulder pose is not high than elbow: shoulder.y=%f elbow.y=%f" % (shoulder.y, elbow.y)

	"""
	# 肘在肩膀和手腕的同侧
	if (elbow.x < shoulder.x) != (elbow.x < wrist.x):
		print(now_time_str(), "# 判断身体部位是否合理：右侧水平方向不合理：elbow.x=%f shoulder.x=%f wrist.x=%f" % (elbow.x, shoulder.x, wrist.x))
		return False, "Right elbow pose is not right than shoulder and wrist: elbow.x=%f shoulder.x=%f wrist.x=%f" % (elbow.x, shoulder.x, wrist.x)
	"""
	# 右侧肘在肩膀右侧
	if (elbow.x > shoulder.x):
		print(now_time_str(), "# 判断身体部位是否合理：右侧肘不在肩膀右侧：elbow.x=%f shoulder.x=%f" % (elbow.x, shoulder.x))
		return False, "Right elbow pose is not right than shoulder: elbow.x=%f shoulder.x=%f" % (elbow.x, shoulder.x)

	# left
	shoulder = pose_landmarks[11]
	elbow = pose_landmarks[13]
	# wrist = pose_landmarks[15]

	"""
	# 肩膀高于肘和手腕
	if (shoulder.y > elbow.y) != (shoulder.y > wrist.y):
		print(now_time_str(), "# 判断身体部位是否合理：左侧高度不合理：shoulder.y=%f elbow.y=%f wrist.y=%f" % (shoulder.y, elbow.y, wrist.y))
		return False, "Left shoulder pose is not high than elbow and wrist: shoulder.y=%f elbow.y=%f wrist.y=%f" % (shoulder.y, elbow.y, wrist.y)
	"""
	# 肩膀高于肘和手腕
	if (shoulder.y > elbow.y):
		print(now_time_str(), "# 判断身体部位是否合理：左侧高度不合理：shoulder.y=%f elbow.y=%f " % (shoulder.y, elbow.y))
		return False, "Left shoulder pose is not high than elbow: shoulder.y=%f elbow.y=%f" % (shoulder.y, elbow.y)

	"""
	# 肘在肩膀和手腕的同侧
	if (elbow.x < shoulder.x) != (elbow.x < wrist.x):
		print(now_time_str(), "# 判断身体部位是否合理：左侧侧水平方向不合理：elbow.x=%f shoulder.x=%f wrist.x=%f" % (elbow.x, shoulder.x, wrist.x))
		return False, "Left elbow pose is not left than shoulder and wrist: elbow.x=%f shoulder.x=%f wrist.x=%f" % (elbow.x, shoulder.x, wrist.x)
	"""
	# 左侧肘在肩膀左侧
	if (elbow.x < shoulder.x):
		print(now_time_str(), "# 判断身体部位是否合理：左侧肘不在肩膀左侧：elbow.x=%f shoulder.x=%f" % (elbow.x, shoulder.x))
		return False, "Left elbow pose is not left than shoulder: elbow.x=%f shoulder.x=%f" % (elbow.x, shoulder.x)

	print(now_time_str(), "# 判断身体部位是否合理：身体姿态合理")
	return True, ""


count_shoulder = 0
count_eye_shoulder = 0
count_eye_elbow = 0
count_no_person = 0
count_face = 0

pose_detector = None
face_detector = None
segmenter_detector = None
object_detector = None
alarm_max = 5


def do_detect(mp_image: mp.Image, frame_tms):
	global count_shoulder, count_eye_shoulder, count_eye_elbow, alarm_max, count_no_person, count_face
	global pose_detector, face_detector, segmenter_detector, object_detector

	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	object_detect_result = object_detector.detect_for_video(mp_image, frame_tms)
	find_person = False
	for obj in object_detect_result.detections:
		for c in obj.categories:
			if c.category_name == 'person':
				find_person = True
				break
	object_detect_result = None

	if not find_person:
		count_shoulder = 0
		count_eye_elbow = 0
		count_eye_shoulder = 0
		count_face = 0
		count_no_person = alarm_max
		print(now_time_str(), "# 没有检测人(%d)" % (count_no_person))
		return
	if count_no_person > 0:
		print(now_time_str(), "# 检测到人出现(%d)" % (count_no_person))
		count_no_person = count_no_person - 1
	elif count_no_person == 0:
		# set to -1, keep person exist
		count_no_person = -1
		print(now_time_str(), "# 检测到人持续出现")
		if in_time(6, 50, 7, 30) or in_time(17, 0, 19, 00):
			subprocess.call(["ffplay", "-volume", "40", "-nodisp", "-autoexit", str(get_script_path() / 'audio' / 'mirror.mp3')])

	# 判断是否存在人脸
	face_detect_result = face_detector.detect_for_video(mp_image, frame_tms)
	if len(face_detect_result.detections) < 1:
		print(now_time_str(), "# 没有检测到人脸信息")
		face_detect_result = None
		mp_image = None
		count_face = 0
		return
	face_detect_result = None

	count_face = count_face + 1
	print(now_time_str(), "# 检测到人脸信息%d次" % (count_face))
	if count_face >= 360:
		if (count_face - 360) % 10 == 0:
			subprocess.call(["ffplay", "-nodisp", "-autoexit", str(get_script_path() / 'audio' / 'relax.mp3')])

	pose_detect_result = pose_detector.detect_for_video(mp_image, frame_tms)

	# 判断是否检测到人的关键部位
	if len(pose_detect_result.pose_landmarks) < 1:
		print(now_time_str(), "# 没有检测到姿态信息")
		pose_detect_result = None
		mp_image = None
		return

	# 只判断一个人的关键部位
	pose_landmarks = pose_detect_result.pose_landmarks[0]

	# 判断身体关键部位的位置是否合理
	(ok_pose, pose_detail) = isok_pose(pose_landmarks)
	if not ok_pose:
		"""
		img_path = get_script_path() / 'imgs' / Path(now + '.jpg')
		annotated_image = draw_landmarks_on_image(mp_image, pose_detect_result)
		draw_text_on_image(annotated_image, now + " " + pose_detail)
		cv2.imwrite(str(img_path), annotated_image)
		annotated_image = None
		"""
		print(now_time_str(), "# 姿态信息中的关键部位位置不合理", pose_detail)
		pose_detect_result = None
		mp_image = None
		return

	# 判断肩膀的倾斜角度
	if len(pose_landmarks) < 13:
		pose_detect_result = None
		mp_image = None
		return
	shoulder_degree = calc_degree(pose_landmarks[12], pose_landmarks[11])
	if shoulder_degree > 15.0:
		count_shoulder = count_shoulder + 1
		print(now_time_str(), "! 肩膀倾斜度数%d，第%d次" % (int(shoulder_degree), count_shoulder))
		if count_shoulder >= alarm_max:
			print(now_time_str(), "! 肩膀倾斜告警，已达到%d次" % (count_shoulder))
			count_shoulder = 0
			img_path = get_script_path() / 'imgs' / Path(now_time_str() + '.jpg')
			annotated_image = draw_landmarks_on_image(mp_image, pose_detect_result)
			draw_text_on_image(annotated_image, "%s Shoulder degree %d" % (now_time_str(), shoulder_degree))
			cv2.imwrite(str(img_path), annotated_image)
			annotated_image = None
			subprocess.call(["ffplay", "-volume", "40", "-nodisp", "-autoexit", str(get_script_path() / 'audio' / 'shoulder.mp3')])
	else:
		count_shoulder = 0
	print(now_time_str(), "# 肩膀倾斜度数", int(shoulder_degree))

	"""
	right_eye_shoulder_degree = calc_degree(pose_landmarks[12], pose_landmarks[6])
	if right_eye_shoulder_degree < 60.0:
		count_eye_shoulder = count_eye_shoulder + 1
		print(now_time_str(), "! 低头，眼肩度数%d），第%d次" % (int(right_eye_shoulder_degree), count_eye_shoulder))
		if count_eye_shoulder >= alarm_max:
			print(now_time_str(), "! 低头告警，已达到%d次" % (count_eye_shoulder))
			count_eye_shoulder = 0
			img_path = get_script_path() / 'imgs' / Path(now_time_str() + '.jpg')
			annotated_image = draw_landmarks_on_image(mp_image, pose_detect_result);
			draw_text_on_image(annotated_image, now_time_str() + " Right eye and shoulder degree")
			cv2.imwrite(str(img_path), annotated_image)
			annotated_image = None
			subprocess.call(["ffplay", "-nodisp", "-autoexit", str(get_script_path() / 'audio' / 'eye.mp3')])
	else:
		count_eye_shoulder = 0
	print(now_time_str(), "# 右眼肩度数", int(right_eye_shoulder_degree))
	"""

	if len(pose_landmarks) < 15:
		pose_detect_result = None
		mp_image = None
		return
	right_eye_elbow_degree = calc_degree(pose_landmarks[14], pose_landmarks[6])
	left_eye_elbow_degree = calc_degree(pose_landmarks[13], pose_landmarks[3])
	# if right_eye_elbow_degree < 50.0 and left_eye_elbow_degree < 50.0:
	if (right_eye_elbow_degree + left_eye_elbow_degree < 115.0) or (right_eye_elbow_degree < 60.0):
		count_eye_elbow = count_eye_elbow + 1
		print(now_time_str(), "! 低头，眼肘度数(L%d，R%d)，第%d次" % (int(left_eye_elbow_degree), int(right_eye_elbow_degree), count_eye_elbow))
		if count_eye_elbow >= alarm_max:
			print(now_time_str(), "! 低头告警，已达到%d次" % (count_eye_shoulder))
			count_eye_elbow = 0
			img_path = get_script_path() / 'imgs' / Path(now_time_str() + '.jpg')
			annotated_image = draw_landmarks_on_image(mp_image, pose_detect_result)
			draw_text_on_image(annotated_image, "%s Right eye elbow degree(L:%d, R:%d)" % (now_time_str(), int(left_eye_elbow_degree), int(right_eye_elbow_degree)))
			cv2.imwrite(str(img_path), annotated_image)
			annotated_image = None
			subprocess.call(["ffplay", "-volume", "40", "-nodisp", "-autoexit", str(get_script_path() / 'audio' / 'eye.mp3')])
	else:
		count_eye_elbow = 0
	print("%s# 眼肘度数(L:%d, R:%d)" % (now_time_str(), int(left_eye_elbow_degree), int(right_eye_elbow_degree)))

	pose_detect_result = None
	mp_image = None

	"""
	print("Left Eye Shoulder Degree: ", calc_degree(pose_landmarks[11], pose_landmarks[3]))
	print("Left Eye Elbow Degree: ", calc_degree(pose_landmarks[13], pose_landmarks[3]))
	print("Eye Degree: ", calc_degree(pose_landmarks[6], pose_landmarks[3]))
	"""


def run() -> None:
	global pose_detector, face_detector, segmenter_detector, object_detector

	# Start capturing video input from the camera
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
	# cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1) # Relative position of the video file 0-start 1-end
	# cap.set(cv2.CAP_PROP_FPS, 10) # 限制帧率为1FPS
	# cap.set(cv2.CAP_PROP_FRAME_COUNT, 1) # 设置视频文件的帧数

	pose_detector = get_pose_detector()
	face_detector = get_face_detector()
	# segmenter_detector = get_segmenter_detector()
	object_detector = get_object_detector()

	# Continuously capture images from the camera and run inference
	while cap.isOpened():
		global sigterm
		if sigterm:
			break

		time.sleep(1)
		success, image = cap.read()
		if not success:
			continue

		# Convert the image from BGR to RGB as required by the TFLite model.
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = None
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
		rgb_image = None

		do_detect(mp_image, time.time_ns() // 1_000_000)

		mp_image = None

	print("退出循环，关闭资源")
	pose_detector.close()
	face_detector.close()
	# segmenter_detector.close()
	object_detector.close()

	cap.release()
	cv2.destroyAllWindows()


# 姿势识别
def get_pose_detector():
	BaseOptions = mp.tasks.BaseOptions
	PoseLandmarker = mp.tasks.vision.PoseLandmarker
	PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	# Create a pose landmarker instance with the video mode:
	base_options = python.BaseOptions(
		model_asset_path=str(get_script_path() / 'pose_landmarker.task'), 
		delegate=python.BaseOptions.Delegate.CPU)
	options = PoseLandmarkerOptions(
		base_options=base_options,
    	running_mode=VisionRunningMode.VIDEO)

	return PoseLandmarker.create_from_options(options)


# 人脸识别
def get_face_detector():
	BaseOptions = mp.tasks.BaseOptions
	FaceDetector = mp.tasks.vision.FaceDetector
	FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	# Create a face detector instance with the image mode:
	options = FaceDetectorOptions(
		base_options=BaseOptions(model_asset_path=str(get_script_path() / 'blaze_face_short_range.tflite'), delegate=python.BaseOptions.Delegate.CPU),
		running_mode=VisionRunningMode.VIDEO)
	return FaceDetector.create_from_options(options)


# 图片分割
def get_segmenter_detector():
	InteractiveSegmenter = mp.tasks.vision.InteractiveSegmenter
	VisionRunningMode = mp.tasks.vision.RunningMode
	# Create a image segmenter instance with the image mode:
	# Create the options that will be used for InteractiveSegmenter
	base_options = python.BaseOptions(model_asset_path=str(get_script_path() / 'magic_touch.tflite'), delegate=python.BaseOptions.Delegate.CPU)
	options = vision.ImageSegmenterOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE, output_category_mask=True)
	return InteractiveSegmenter.create_from_options(options)


def get_object_detector():
	import mediapipe as mp

	BaseOptions = mp.tasks.BaseOptions
	ObjectDetector = mp.tasks.vision.ObjectDetector
	ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	options = ObjectDetectorOptions(
		base_options=BaseOptions(model_asset_path=str(get_script_path() / 'efficientdet_lite0.tflite'), delegate=python.BaseOptions.Delegate.CPU),
		max_results=5,
		running_mode=VisionRunningMode.VIDEO)

	return ObjectDetector.create_from_options(options)


def main():
	run()


if __name__ == '__main__':
	main()

import cv2
import dlib
import face_recognition
from mtcnn import MTCNN
from retinaface import RetinaFace

from enum import Enum
import time
import logging
import os, shutil

class FaceDetectionLibrary(Enum):
    OPEN_CV = 1
    DLIB = 2
    FACE_RECOGNITION_LIB = 3
    MTCNN = 4
    RETINA_FACE = 5

TEST_IMAGE_NO_FACE_PATH = 'testing/data/test_image_no_face.jpg'
TEST_IMAGE_SINGLE_NO_MASK_PATH = 'testing/data/test_image_single_no_mask.jpg'
TEST_IMAGE_SINGLE_MASKED_PATH = 'testing/data/test_image_single_masked.jpg'
TEST_IMAGE_MULTIPLE_NO_MASK_PATH = 'testing/data/test_image_multiple_no_mask.jpg'
TEST_IMAGE_MULTIPLE_MASKED_PATH = 'testing/data/test_image_multiple_masked.jpg'
TEST_IMAGE_MIXED_PATH = 'testing/data/test_image_mixed.jpg'

TEST_VIDEO_NO_FACE_PATH = 'testing/data/test_video_no_face.mp4'
TEST_VIDEO_SINGLE_NO_MASK_PATH = 'testing/data/test_video_single_no_mask.mp4'
TEST_VIDEO_SINGLE_MASKED_PATH = 'testing/data/test_video_single_masked.mp4'
TEST_VIDEO_MULTIPLE_NO_MASK_PATH = 'testing/data/test_video_multiple_no_mask.mp4'
TEST_VIDEO_MULTIPLE_MASKED_PATH = 'testing/data/test_video_multiple_masked.mp4'
TEST_VIDEO_MIXED_PATH = 'testing/data/test_video_mixed.mp4'

TEST_SAVE_PATH = 'testing/results'

if os.path.exists('testing/results/results.log'):
    os.remove('testing/results/results.log')

logger = logging.getLogger(__name__)
logging.basicConfig(filename = 'testing/results/results.log', encoding = 'utf-8', level = logging.INFO)

def main():
    # test_open_cv()
    # test_dlib()
    # test_face_recognition_lib()
    # test_mtcnn()
    test_retina_face()

def clear_folder(folder):
    if not os.path.exists(folder):
        return
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error('Failed to delete %s. Reason: %s' % (file_path, e))

# Testing libraries' face detection

def test_open_cv():
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('Testing face detection using OpenCV library')

    clear_folder(TEST_SAVE_PATH + '/open_cv')

    start_time = time.time()
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_NO_FACE_PATH, TEST_SAVE_PATH + '/open_cv/test_image_no_face_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/open_cv/test_image_single_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/open_cv/test_image_single_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/open_cv/test_image_multiple_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/open_cv/test_image_multiple_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.OPEN_CV, TEST_IMAGE_MIXED_PATH, TEST_SAVE_PATH + '/open_cv/test_image_mixed_result.jpg')
    
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_NO_FACE_PATH, TEST_SAVE_PATH + '/open_cv/test_video_no_face_result.avi')
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/open_cv/test_video_single_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/open_cv/test_video_single_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/open_cv/test_video_multiple_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/open_cv/test_video_multiple_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.OPEN_CV, TEST_VIDEO_MIXED_PATH, TEST_SAVE_PATH + '/open_cv/test_video_mixed_result.avi')
    end_time = time.time()

    logger.info(f'\n -Overall time taken: {end_time - start_time:0.4f}s')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

def test_dlib():
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('Testing face detection using Dlib library')

    clear_folder(TEST_SAVE_PATH + '/dlib')

    start_time = time.time()
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_NO_FACE_PATH, TEST_SAVE_PATH + '/dlib/test_image_no_face_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/dlib/test_image_single_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/dlib/test_image_single_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/dlib/test_image_multiple_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/dlib/test_image_multiple_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.DLIB, TEST_IMAGE_MIXED_PATH, TEST_SAVE_PATH + '/dlib/test_image_mixed_result.jpg')
    
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_NO_FACE_PATH, TEST_SAVE_PATH + '/dlib/test_video_no_face_result.avi')
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/dlib/test_video_single_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/dlib/test_video_single_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/dlib/test_video_multiple_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/dlib/test_video_multiple_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.DLIB, TEST_VIDEO_MIXED_PATH, TEST_SAVE_PATH + '/dlib/test_video_mixed_result.avi')
    end_time = time.time()

    logger.info(f'\n -Overall time taken: {end_time - start_time:0.4f}s')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

def test_face_recognition_lib():
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('Testing face detection using face_recognition library')

    clear_folder(TEST_SAVE_PATH + '/face_recognition_lib')

    start_time = time.time()
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_NO_FACE_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_no_face_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_single_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_single_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_multiple_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_multiple_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_IMAGE_MIXED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_image_mixed_result.jpg')
    
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_NO_FACE_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_no_face_result.avi')
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_single_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_single_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_multiple_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_multiple_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.FACE_RECOGNITION_LIB, TEST_VIDEO_MIXED_PATH, TEST_SAVE_PATH + '/face_recognition_lib/test_video_mixed_result.avi')
    end_time = time.time()

    logger.info(f'\n -Overall time taken: {end_time - start_time:0.4f}s')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

def test_mtcnn():
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('Testing face detection using MTCNN library')

    clear_folder(TEST_SAVE_PATH + '/mtcnn')

    start_time = time.time()
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_NO_FACE_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_no_face_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_single_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_single_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_multiple_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_multiple_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.MTCNN, TEST_IMAGE_MIXED_PATH, TEST_SAVE_PATH + '/mtcnn/test_image_mixed_result.jpg')
    
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_NO_FACE_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_no_face_result.avi')
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_single_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_single_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_multiple_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_multiple_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.MTCNN, TEST_VIDEO_MIXED_PATH, TEST_SAVE_PATH + '/mtcnn/test_video_mixed_result.avi')
    end_time = time.time()

    logger.info(f'\n -Overall time taken: {end_time - start_time:0.4f}s')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

def test_retina_face():
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('Testing face detection using RetinaFace library')

    clear_folder(TEST_SAVE_PATH + '/retina_face')

    start_time = time.time()
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_NO_FACE_PATH, TEST_SAVE_PATH + '/retina_face/test_image_no_face_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/retina_face/test_image_single_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/retina_face/test_image_single_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/retina_face/test_image_multiple_no_mask_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/retina_face/test_image_multiple_masked_result.jpg')
    test_face_detection_image(FaceDetectionLibrary.RETINA_FACE, TEST_IMAGE_MIXED_PATH, TEST_SAVE_PATH + '/retina_face/test_image_mixed_result.jpg')
    
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_NO_FACE_PATH, TEST_SAVE_PATH + '/retina_face/test_video_no_face_result.avi')
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_SINGLE_NO_MASK_PATH, TEST_SAVE_PATH + '/retina_face/test_video_single_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_SINGLE_MASKED_PATH, TEST_SAVE_PATH + '/retina_face/test_video_single_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_MULTIPLE_NO_MASK_PATH, TEST_SAVE_PATH + '/retina_face/test_video_multiple_no_mask_result.avi')
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_MULTIPLE_MASKED_PATH, TEST_SAVE_PATH + '/retina_face/test_video_multiple_masked_result.avi')
    test_face_detection_video(FaceDetectionLibrary.RETINA_FACE, TEST_VIDEO_MIXED_PATH, TEST_SAVE_PATH + '/retina_face/test_video_mixed_result.avi')
    end_time = time.time()

    logger.info(f'\n -Overall time taken: {end_time - start_time:0.4f}s')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# Util
    
def test_face_detection_image(face_detection_lib, image_file_path, save_file_name):
    test_image = cv2.imread(image_file_path)
    start_time = time.time()

    if face_detection_lib == FaceDetectionLibrary.OPEN_CV:
        test_image = detect_faces_open_cv(test_image)
    elif face_detection_lib == FaceDetectionLibrary.DLIB:
        test_image = detect_faces_dlib(test_image)
    elif face_detection_lib == FaceDetectionLibrary.FACE_RECOGNITION_LIB:
        test_image = detect_faces_face_recognition_lib(test_image)
    elif face_detection_lib == FaceDetectionLibrary.MTCNN:
        test_image = detect_faces_mtcnn(test_image)
    elif face_detection_lib == FaceDetectionLibrary.RETINA_FACE:
        test_image = detect_faces_retina_face(test_image)

    end_time = time.time()
    cv2.imwrite(save_file_name, test_image)

    logger.info(f' -Running face detection for {image_file_path} took: {end_time - start_time:0.4f}s')
    
def test_face_detection_video(face_detection_lib, video_file_path, save_file_name):
    test_video = cv2.VideoCapture(video_file_path)
    frame_width = int(test_video.get(3)) 
    frame_height = int(test_video.get(4)) 
    size = (frame_width, frame_height) 

    result_writer = cv2.VideoWriter(save_file_name,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        30, size)  
    start_time = time.time()
    
    while test_video.isOpened():
        result, video_frame = test_video.read()
        if result is False:
            break

        if face_detection_lib == FaceDetectionLibrary.OPEN_CV:
            video_frame = detect_faces_open_cv(video_frame)
        elif face_detection_lib == FaceDetectionLibrary.DLIB:
            video_frame = detect_faces_dlib(video_frame)
        elif face_detection_lib == FaceDetectionLibrary.FACE_RECOGNITION_LIB:
            video_frame = detect_faces_face_recognition_lib(video_frame)
        elif face_detection_lib == FaceDetectionLibrary.MTCNN:
            video_frame = detect_faces_mtcnn(video_frame)
        elif face_detection_lib == FaceDetectionLibrary.RETINA_FACE:
            video_frame = detect_faces_retina_face(video_frame)

        result_writer.write(video_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()

    test_video.release()
    result_writer.release()

    logger.info(f' -Running face detection for {video_file_path} took: {end_time - start_time:0.4f}s')
    
def detect_faces_open_cv(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor = 1.1, minNeighbors = 10, minSize = (20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return image

def detect_faces_dlib(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def detect_faces_face_recognition_lib(image):
    # Using HOG-based model
    # faces = face_recognition.face_locations(image)

    # Using a pre-trained CNN
    faces = face_recognition.face_locations(image, model = "cnn")

    for (top, right, bottom, left) in faces:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    return image

def detect_faces_mtcnn(image):
    detector = MTCNN()

    detected_faces = detector.detect_faces(image)

    for face in detected_faces:
        (x, y, w, h) = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def detect_faces_retina_face(image):
    detected_faces = RetinaFace.detect_faces(image)

    for face in detected_faces.values():
        (top, right, bottom, left) = face['facial_area']        
        cv2.rectangle(image, (bottom, left), (top, right), (0, 255, 0), 2)
    
    return image

if __name__ == '__main__':
    main()
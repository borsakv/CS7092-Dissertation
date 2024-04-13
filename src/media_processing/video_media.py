import cv2
from media_processing.processor import Processor, ObfuscationType

class VideoMedia(Processor):
    def __init__(self, video_file_path: str, save_path: str, obfuscation_level: int, obfuscation_type: ObfuscationType, fps: int = 30) -> None:
        super().__init__(obfuscation_level, obfuscation_type)

        self.video_file_path = video_file_path
        self.save_path = save_path
        self.fps = fps

    def read_video(self) -> None:
        self.video = cv2.VideoCapture(self.video_file_path)
        frame_width = int(self.video.get(3)) 
        frame_height = int(self.video.get(4)) 
        self.size = (frame_width, frame_height) 

    def obfuscate_video_faces(self) -> None:
        self.read_video()

        result_writer = cv2.VideoWriter(self.save_path,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        self.fps, self.size)    
        
        while self.video.isOpened():
            result, video_frame = self.video.read()
            if result is False:
                break
            
            video_frame = self.obfuscate_faces(video_frame)
            result_writer.write(video_frame) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        result_writer.release()

    def obfuscate_full_video(self) -> None:
        self.read_video()
        result_writer = cv2.VideoWriter(self.save_path,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        self.fps, self.size) 
        
        while self.video.isOpened():
            result, video_frame = self.video.read()
            if result is False:
                break

            video_frame = self.obfuscate_everything(video_frame)
            result_writer.write(video_frame) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        result_writer.release()
import cv2
import torch
from pathlib import Path
import logging
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from typing import Union, Tuple

class VideoContentAnalyzer:
    def __init__(self, model_name: str = "Falconsai/nsfw_image_detection"):
        """
        Initialize the content analyzer with the NSFW detection model.
        
        Parameters:
            model_name (str): The name or path of the pre-trained model.
        """
        # Select device: use GPU if available, else CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the pre-trained image classification model and move it to the selected device.
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        # Load the corresponding image processor.
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Set up logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_frame(self, image: Image.Image) -> Tuple[str, float]:
        """
        Analyze a single frame for NSFW content.
        
        Parameters:
            image (PIL.Image.Image): The image to analyze.
            
        Returns:
            Tuple[str, float]: A tuple containing the prediction label and confidence score.
        """
        with torch.no_grad():
            # Process the image to the required tensor format.
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # Perform inference using the model.
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities.
            probs = torch.softmax(logits, dim=-1)
            # Find the index of the highest scoring prediction.
            pred_idx = logits.argmax(-1).item()
            confidence = probs[0][pred_idx].item()
            
            # Retrieve the human-readable label from the model's configuration.
            return self.model.config.id2label[pred_idx], confidence

    def analyze_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 6,
        stop_on_nsfw: bool = True,
        nsfw_threshold: float = 0.5
    ) -> dict:
        """
        Analyze frames from a video for NSFW content.
        
        Parameters:
            video_path (Union[str, Path]): Path to the video file.
            num_frames (int): Number of frames to analyze.
            stop_on_nsfw (bool): Whether to stop analysis when NSFW content is detected.
            nsfw_threshold (float): Confidence threshold for NSFW classification.
            
        Returns:
            dict: Analysis results including frame classifications and overall verdict.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting analysis of video: {video_path}")
        
        # Open the video file.
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total_frames // num_frames)
            
            results = {
                'frames_analyzed': 0,
                'nsfw_detected': False,
                'frame_results': [],
                'first_nsfw_frame': None
            }

            # Loop through and analyze selected frames.
            for i in range(num_frames):
                frame_position = min(i * interval, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame at position {frame_position}")
                    continue

                # Convert the frame from BGR (OpenCV default) to RGB.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Create a PIL Image from the frame.
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze the frame using the NSFW detection model.
                label, confidence = self.analyze_frame(pil_image)
                
                frame_result = {
                    'frame_number': i + 1,
                    'position': frame_position,
                    'prediction': label,
                    'confidence': confidence
                }
                results['frame_results'].append(frame_result)
                results['frames_analyzed'] += 1

                self.logger.info(
                    f"Frame {i+1}/{num_frames} Analysis: "
                    f"Classification: {label}, Confidence: {confidence:.2%}"
                )

                # If NSFW content is detected above the threshold, record and optionally stop.
                if label.lower() == "nsfw" and confidence >= nsfw_threshold:
                    results['nsfw_detected'] = True
                    results['first_nsfw_frame'] = frame_result
                    if stop_on_nsfw:
                        self.logger.warning(
                            f"NSFW content detected in frame {i+1} "
                            f"with {confidence:.2%} confidence. Stopping analysis."
                        )
                        break

            return results

        finally:
            cap.release()

# Example usage:
if __name__ == "__main__":
    try:
        # Initialize the video content analyzer.
        analyzer = VideoContentAnalyzer()
        
        # Specify the path to your video file.
        video_file = Path("C:/test1.mp4")  # Adjust this path as needed.
        
        # Analyze the video.
        results = analyzer.analyze_video(
            video_path=video_file,
            num_frames=6,
            stop_on_nsfw=True,
            nsfw_threshold=0.5
        )
        
        # Display the overall analysis result.
        if results['nsfw_detected']:
            nsfw_frame = results['first_nsfw_frame']
            print(f"\nNSFW content detected!")
            print(f"First occurrence: Frame {nsfw_frame['frame_number']}")
            print(f"Confidence: {nsfw_frame['confidence']:.2%}")
        else:
            print(f"\nNo NSFW content detected in {results['frames_analyzed']} analyzed frames")
            
        # Print detailed analysis for each frame.
        print("\nDetailed frame analysis:")
        for frame in results['frame_results']:
            print(f"Frame {frame['frame_number']}: "
                  f"{frame['prediction']} ({frame['confidence']:.2%} confidence)")
            
    except Exception as e:
        logging.error(f"Error during video analysis: {str(e)}")

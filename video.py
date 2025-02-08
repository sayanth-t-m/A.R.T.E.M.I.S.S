import cv2
import os
from pathlib import Path
import logging
from typing import Union, Optional

def extract_frames(
    video_path: Union[str, Path],
    output_folder: Union[str, Path] = "frames",
    num_frames: int = 6,
    output_format: str = "jpg",
    quality: int = 95
) -> bool:
    """
    Extracts a specified number of frames from the input video at equal intervals.

    Parameters:
        video_path (Union[str, Path]): Path to the video file
        output_folder (Union[str, Path]): Folder to save the extracted frames
        num_frames (int): Number of frames to extract (default: 6)
        output_format (str): Format to save frames ('jpg' or 'png', default: 'jpg')
        quality (int): Image quality for JPEG format (1-100, default: 95)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Convert paths to Path objects for better cross-platform compatibility
        video_path = Path(video_path)
        output_folder = Path(output_folder)

        # Input validation
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if num_frames < 1:
            raise ValueError("Number of frames must be positive")
            
        if quality < 1 or quality > 100:
            raise ValueError("Quality must be between 1 and 100")
            
        if output_format.lower() not in ['jpg', 'jpeg', 'png']:
            raise ValueError("Output format must be 'jpg' or 'png'")

        # Create the output folder
        output_folder.mkdir(parents=True, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, "
                   f"Duration: {duration:.2f} seconds")

        # Adjust number of frames if video is too short
        if total_frames < num_frames:
            logger.warning(
                f"Video has only {total_frames} frames. Adjusting request from "
                f"{num_frames} to {total_frames} frames."
            )
            num_frames = total_frames

        # Calculate frame interval
        interval = max(1, total_frames // num_frames)
        
        # Extract frames
        frames_saved = 0
        for i in range(num_frames):
            frame_position = min(i * interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if ret:
                # Prepare filename
                frame_filename = output_folder / f"frame_{i+1:03d}.{output_format}"
                
                # Save frame with appropriate parameters
                if output_format.lower() in ['jpg', 'jpeg']:
                    cv2.imwrite(str(frame_filename), frame, 
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:  # PNG
                    cv2.imwrite(str(frame_filename), frame)
                    
                frames_saved += 1
                logger.info(f"Saved frame {frames_saved}/{num_frames}: {frame_filename}")
            else:
                logger.warning(f"Failed to read frame at position {frame_position}")

        cap.release()
        
        if frames_saved == 0:
            raise RuntimeError("No frames were successfully extracted")
            
        logger.info(f"Successfully extracted {frames_saved} frames")
        return True

    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return False

# Example usage:
if __name__ == "__main__":
    # Use Path for better cross-platform compatibility
    video_file = Path("C:/video_1.mp4")
    success = extract_frames(
        video_file,
        output_folder="extracted_frames",
        num_frames=6,
        output_format="jpg",
        quality=95
    )
    
    if not success:
        print("Frame extraction failed. Check the logs for details.")
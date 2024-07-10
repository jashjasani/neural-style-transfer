import os
import subprocess
import shutil
from glob import glob

def create_video_from_intermediate_results(results_path, img_format=None, fps=60, frame_step=1):
    out_file_name = 'out.mp4'
    first_frame = 0

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        # If img_format is not specified, try to detect it
        if img_format is None:
            image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
            for ext in image_extensions:
                if glob(os.path.join(results_path, f'*.{ext}')):
                    img_format = ext
                    break
            if img_format is None:
                print(f"No image files found in {results_path}")
                return

        # Get all image files in the directory
        image_files = sorted(glob(os.path.join(results_path, f'*.{img_format}')))
        
        if not image_files:
            print(f"No {img_format} files found in {results_path}")
            return

        # Use only every frame_step-th frame
        image_files = image_files[::frame_step]
        number_of_frames_to_process = len(image_files)
        
        # Create a temporary directory for symlinks with proper naming
        temp_dir = os.path.join(results_path, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create symlinks with proper naming
        for i, img_file in enumerate(image_files):
            os.symlink(img_file, os.path.join(temp_dir, f'{i:04d}.{img_format}'))

        pattern = os.path.join(temp_dir, f'%04d.{img_format}')
        out_video_path = os.path.join(results_path, out_file_name)

        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        input_options = ['-r', str(fps), '-i', pattern]
        encoding_options = ['-c:v', 'libx264', '-preset', 'faster', '-crf', '23', '-pix_fmt', 'yuv420p']
        
        try:
            subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, out_video_path])
            print(f"Video created successfully: {out_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while creating video: {e}")
        finally:
            # Clean up temporary directory
            for file in os.listdir(temp_dir):
                os.unlink(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
    else:
        print(f'{ffmpeg} not found in the system path, aborting.')

# Usage
create_video_from_intermediate_results("/teamspace/studios/this_studio/pytorch-neural-style-transfer/data/output-images", fps=60, frame_step=2)
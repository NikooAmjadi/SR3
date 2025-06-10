import os
from PIL import Image

def process_folder(folder_name):
    hr_dimen = 256
    lr_dimen = 128
    input_folder = os.path.join(folder_name, "org")
    output_hr = os.path.join(folder_name, f"hr_{hr_dimen}")
    output_lr = os.path.join(folder_name, f"lr_{lr_dimen}")

    os.makedirs(output_hr, exist_ok=True)
    os.makedirs(output_lr, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    for i, filename in enumerate(image_files, start=1):
        try:
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                w, h = img.size
                side = min(w, h)

                # Center crop
                left = (w - side) // 2
                top = (h - side) // 2
                right = left + side
                bottom = top + side
                img_cropped = img.crop((left, top, right, bottom))

                # Resize
                img_hr = img_cropped.resize((hr_dimen, hr_dimen), Image.BICUBIC)
                img_lr = img_cropped.resize((lr_dimen, lr_dimen), Image.BICUBIC)

                # Save both HR and LR
                filename_out = f"image_{i:03d}.png"
                img_hr.save(os.path.join(output_hr, filename_out))
                img_lr.save(os.path.join(output_lr, filename_out))
                print(f"Saved {filename_out} to {output_hr} and {output_lr}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Run for both train and test folders
process_folder("")

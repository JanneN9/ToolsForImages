# ToolsForImages

Codes made by Claude

1) Remove duplicate
   - download several web pages from Civitai
   - result will single instance of upload images
   - all folder under main folder is scanned
  
Image deduplication and renaming tool.

- Scans a folder recursively for images
- Detects exact duplicates using MD5 hash
- Deletes duplicates (keeps one copy)
- Renames all remaining images to sequential numbers (001.jpg, 002.jpg, ...)
  into a flat output folder

Usage:
    python image_dedup.py <input_folder> <output_folder>

Example:
    python image_dedup.py ./my_images ./cleaned_images

  
2) Image matcher
   - folder of small thumbnails
   - search originals from folders
  
<img width="1434" height="908" alt="image" src="https://github.com/user-attachments/assets/3f854849-97c9-4c53-ae89-fd096921f434" />

<img width="399" height="150" alt="image" src="https://github.com/user-attachments/assets/aea5db97-33be-4aad-a84d-d58ef2023206" />

  
3) Sorting generated images
   - publish, save, delete, ...
   - 
<img width="1244" height="834" alt="image" src="https://github.com/user-attachments/assets/98c7bdb8-76a5-406c-8280-4805f0c867ad" />

  
Pair detection — scans the source folder and matches files by removing Own / Pub from the stem (e.g. MyArt_Own_001.png + MyArt_Pub_001.png → key MyArt_001). Each pair shows both thumbnails side-by-side with ✓/✗ indicators for which files exist.

Five one-click action buttons on every card:
<img width="722" height="256" alt="image" src="https://github.com/user-attachments/assets/88e8b3c4-eadd-4833-b7cf-c8a14edf5b63" />




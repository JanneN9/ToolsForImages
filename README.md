# ToolsForImages

Codes made by Claude

1) Remove duplicate
   - download several web pages from Civitai
   - result will single instance of upload images
  
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
  
3) Sorting generated images
   - publish, save, delete, ...
  

# Data

## Test Annotations (`test.json`)

`test.json` contains 360 annotated test samples. Each entry has the following
structure:

```json
{
  "panoramic_image": "3285.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<panoramic_image>\nQuestion: What are the azimuth and elevation ... ?"
    },
    ...
  ],
  "metadata": {
    "object_label": "a painted wild dog in a frame",
    "azimuth_degrees": 8.4,
    "elevation_degrees": 16.2,
    "bbox_diagonal_degrees": 28.5,
    ...
  }
}
```

The `panoramic_image` field contains only the filename. Pass the parent
directory via `--pano_dir` when running `eval.py`.

## Panoramic Images — Matterport3D

The evaluation uses 354 equirectangular panoramic images sampled from the
[Matterport3D](https://niessner.github.io/Matterport/) dataset.

### How to obtain the images

Matterport3D requires an academic license agreement. Follow these steps:

1. Fill in the data license agreement at:
   <https://niessner.github.io/Matterport/>
2. After approval, download the dataset using the official script:
   ```bash
   python download_mp.py --type matterport_skybox_images -o /path/to/matterport3d
   ```
3. The evaluation uses the stitched equirectangular panoramas. Convert or
   locate them in the dataset, naming each file as `{scene_id}.jpg`.

### Alternatively — subset download

If you only need the 354 panoramas used in this benchmark, a curated subset
may be released on HuggingFace Datasets alongside the test annotations.
Check the repository README for an updated download link.

# Pupil detection preview
This [Pupil](https://github.com/pupil-labs/pupil) plugin allows the export of frames during a running recording. On the extracted frames, the 2D detector is applied for getting a rough preview of the detection quality. After the recording, a build-in viewer is able to visualize the gained previews.

## Installation
1. Drop the `preview.py` into the plugin directory
2. Activate both the *Frame Publisher* and the *Preview* plugin
3. (Start a recording - the frames are exported.)

## Configuration
Beside the configuration through the GUI, the parameters utilized by the pupil detector may be specified in a file called `user_settings_preview.json` stored in the settings directory (i.e. `pupil_capture_settings`). Such a file may look like this:
```json
{
	"pupil_size_min": 40,
	"pupil_size_max": 200,
	"coarse_detection": false
}
```

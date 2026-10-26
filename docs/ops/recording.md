# Recording a replay

How a replay is recorded to a video on the Jetson, and how to check the result. The
design decision is in `docs/design/incident-sessions.md`, "Recording the acceptance run".

## On the Jetson (the default)

1. The app runs with replay mode on: `HANDOFF_REPLAY=1 python -m ems_handoff` (or
   `--no-window` from ssh; the recording window is opened separately, below).
2. Start a replay with **Record** ticked on the Replay card, or through the API:

       curl -s -X POST -H 'Content-Type: application/json' \
            -d '{"callset":"CS008","provider":"groq","speed":1.0,"record":true}' \
            http://127.0.0.1:5050/api/replay/start

3. The server opens a Chromium app window on the Jetson display (`HANDOFF_REPLAY_DISPLAY`,
   default `:1`) at exactly 1920x1080 (`HANDOFF_RECORD_WIDTH`/`HEIGHT`) with the page zoomed 1.5x
   (`HANDOFF_RECORD_SCALE`), so the layout is the 1280x720 one at twice the pixel density; strips
   the window manager's title bar, and measures the page's client area. Just before burst 1 it
   starts one ffmpeg process that captures that area at 15 fps together with the null sink's
   monitor (`handoff_replay.monitor`), both with
   wall-clock timestamps, preset ultrafast, no post-mux. After the last burst and the final
   report it stops ffmpeg, writes the summary, runs the sync check and adds a line to
   `~/recordings/INDEX.md` with the verdict column left as `pending` for a person.
4. Output: `~/recordings/<date>_<commit>/<callset>.mp4`, with `<callset>.summary.json` (the
   replay status: timings, transcriptions, session, report) and `<callset>.sync.json` beside it.

Audio: the driver plays each burst's samples into the PipeWire null sink `handoff_replay`
(created with `pw-cli` if missing) at the instant it feeds the file source, started
offset by `HANDOFF_REPLAY_SINK_LEAD_MS` (default -270: the player starts 270 ms after the file
source, calibrated from recordings on the Jetson, 2026-10-25). A 50 ms
1 kHz click is mixed over the first 50 ms of each burst, and the banner's sync marker flashes
white for two frames at the same moment.

## The sync check

    python scripts/check_sync.py ~/recordings/<date>_<commit>/CS008.mp4 --expected 1

Finds the clicks in the audio track and the marker flashes in the video track and prints the
offset per burst (click minus flash). A recording passes only if every burst is within 150 ms.
The flash is quantised to a frame (67 ms at 15 fps), so offsets within about ±35 ms are the
measurement floor.

## Copying the video to a laptop

    scp 'voxai@vox-ai:~/recordings/<date>_<commit>/CS008.mp4' ~/Desktop/

Use the Jetson's LAN address if `vox-ai` does not resolve.

## Alternative: record on a laptop

Not the default. Bind the server to the LAN (`HANDOFF_HOST=0.0.0.0`), open
`http://<jetson>:5050/` in a laptop browser at 1920x1080 zoomed 150%, and capture there. The sink audio
stays on the Jetson, so a laptop recording has no click track and the sync check does not
apply; use OBS with a window capture, or ffmpeg's screen grab for your platform.

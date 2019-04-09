import logging
from multiprocessing import Process, Pipe
from pathlib import Path

import zmq
import cv2
import numpy as np
from pyglui import ui

from plugin import Plugin
from methods import Roi
from zmq_tools import Msg_Receiver
from pupil_detectors import Detector_2D
from vis_eye_video_overlay import get_ellipse_points


class PreviewGenerator:
    class ImageStream:
        class FrameWrapper:
            """
            A tiny wrapper for the layout constrained by the detector.
            """

            def __init__(self, image: np.ndarray):
                self.width = image.shape[1]
                self.height = image.shape[0]
                self.gray = image
                self.timestamp = 0

        FILE_FORMAT = "eye{}_frame{}_confidence{}.png"

        def __init__(
            self, eye_id: int, frame_per_frames: int, folder: Path, frame_size
        ):
            self.frame_per_frames = frame_per_frames
            self.folder = folder
            self.frame_size = frame_size
            self.eye_id = eye_id

            self.__counter = 0
            self.__detector = Detector_2D()

        def add(self, payload) -> bool:
            self.__counter += 1
            if self.__counter % self.frame_per_frames == 0:
                if payload["format"] not in ("gray", "bgr"):
                    raise NotImplementedError(
                        "The eye frame format '{}' is currently not supported!".format(
                            payload["format"]
                        )
                    )

                shape = [self.frame_size[1], self.frame_size[0]]
                if payload["format"] == "bgr":
                    shape.append(3)

                data = payload["__raw_data__"][-1]
                if len(data) == np.prod(shape):
                    raw_frame = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                    grayscale_frame = (
                        raw_frame
                        if len(shape) == 2
                        else cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                    )
                    color_frame = (
                        raw_frame
                        if len(shape) == 3
                        else cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                    )

                    # Extract the pupil
                    pupil_2d = self.__detector.detect(
                        frame_=PreviewGenerator.ImageStream.FrameWrapper(
                            grayscale_frame
                        ),
                        user_roi=Roi(grayscale_frame.shape),
                        visualize=False,
                    )

                    # Visualize the ellipse
                    ellipse = pupil_2d["ellipse"]
                    confidence = pupil_2d["confidence"]
                    if confidence > 0.0:
                        ellipse_points = get_ellipse_points(
                            (ellipse["center"], ellipse["axes"], ellipse["angle"]),
                            num_pts=50,
                        )
                        cv2.polylines(
                            color_frame,
                            [np.asarray(ellipse_points, dtype="i")],
                            True,
                            (0, 0, 255),
                            thickness=1,
                        )

                    # Write the visualization as an image
                    cv2.imwrite(
                        str(
                            self.folder
                            / PreviewGenerator.ImageStream.FILE_FORMAT.format(
                                self.eye_id, self.__counter, confidence
                            )
                        ),
                        color_frame,
                    )

                    return True
                else:
                    raise RuntimeWarning(
                        "Image size {} does not match expected shape.".format(len(data))
                    )

            return False

    def __init__(
        self, url, command_pipe, exception_pipe, frame_per_frames: int, folder: Path
    ):
        if not folder.is_dir():
            raise FileNotFoundError("The given folder does not exists.")

        self.frame_per_frames = frame_per_frames
        self.folder = folder

        self._url = url
        self._command_pipe = command_pipe
        self._status_pipe = exception_pipe

    @staticmethod
    def generate(params: "PreviewGenerator"):
        try:
            # Connect to url and read
            params._status_pipe.send("Connecting to URL '{}'...".format(params._url))
            context = zmq.Context()
            frame_queue = Msg_Receiver(context, params._url, topics=("frame.eye",))
            params._status_pipe.send("Starting receiving frames...")

            streams = {}
            while not params._command_pipe.poll():
                if frame_queue.new_data:
                    topic, payload = frame_queue.recv()
                    id = int(str(topic).split(".")[-1])
                    if id not in streams:
                        streams[id] = PreviewGenerator.ImageStream(
                            eye_id=id,
                            frame_per_frames=params.frame_per_frames,
                            folder=params.folder,
                            frame_size=(payload["width"], payload["height"]),
                        )
                    streams[id].add(payload)

            del frame_queue
        except Exception as e:
            params._status_pipe.send(e)


class Detection_Preview(Plugin):
    icon_chr = "P"
    order = 0.6

    def __init__(
        self,
        g_pool,
        frames_per_frame: int = 120,
        folder: str = None,
    ):
        super().__init__(g_pool)

        logging.info(', '.join(g_pool.__dict__.keys()))

        self.frames_per_frame = frames_per_frame
        self.folder = folder

        command_receiver, self.__command_sender = Pipe(False)
        self.__status_receiver, status_sender = Pipe(False)
        self.__worker = None
        self.__generator = PreviewGenerator(
            url=g_pool.ipc_sub_url,
            command_pipe=command_receiver,
            exception_pipe=status_sender,
            frame_per_frames=frames_per_frame,
            folder=Path(folder),
        )

    def recent_events(self, events):
        if self.__status_receiver.poll():
            status = self.__status_receiver.recv()
            if isinstance(status, Exception):
                raise status
            else:
                logging.info("Image process: {}".format(status))

    def on_notify(self, notification):
        subject = notification["subject"]
        if subject == "recording.started" and self.__worker is None:
            path = notification["rec_path"]
            self.__worker = self.__worker = Process(
                target=PreviewGenerator.generate, args=(self.__generator,), daemon=True
            )

            logging.info("Starting generating previews.")
            self.__worker.start()

        elif (
            subject == "recording.stopped"
            and self.__worker is not None
            and self.__worker.is_alive()
        ):
            logging.info("Stopping generating previews.")
            self.__command_sender.send("exit")
            self.__worker.join(3)
            assert self.__worker.exitcode is not None, "Joining failed."
            self.__worker = None

    def get_init_dict(self):
        return {"frames_per_frame": self.frames_per_frame, "folder": self.folder}

    def clone(self):
        return Detection_Preview(**self.get_init_dict())

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Preview of pupil detection."
        self.menu.append(
            ui.Info_Text(
                "This plugin saves a subset of eye images with their 2D detected ellipses for evaluation purposes."
            )
        )
        self.menu.append(
            ui.Slider(
                "frames_per_frame",
                self,
                min=10,
                step=10,
                max=10000,
                label="Amount of frames skipped",
            )
        )

    def deinit_ui(self):
        self.remove_menu()

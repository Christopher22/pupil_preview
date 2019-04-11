import logging
from multiprocessing import Process, Pipe
from pathlib import Path

import zmq
import cv2
import numpy as np
from pyglui import ui
import glfw

from plugin import Plugin
from methods import Roi
from zmq_tools import Msg_Receiver
from pupil_detectors import Detector_2D
from vis_eye_video_overlay import get_ellipse_points
from pyglui.cygl.utils import draw_gl_texture
from gl_utils import clear_gl_screen, basic_gl_setup, make_coord_system_norm_based

logger = logging.getLogger(__name__)


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

        FILE_FORMAT = "eye{}_frame{}_confidence{:05.4f}.png"

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
                            thickness=2,
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

        def __bool__(self):
            return self.__counter > 0

    def __init__(
        self, url, command_pipe, exception_pipe, frame_per_frames: int, folder: Path
    ):
        if not folder.is_dir():
            raise FileNotFoundError(
                "The given folder '{}' does not exists.".format(folder)
            )

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
            params._status_pipe.send(
                "Starting generating previews and saving them in '{}'...".format(
                    params.folder
                )
            )

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


class PreviewWindow:
    class WindowContextManager:
        def __init__(self, next_handle=None):
            self.__next_handle = next_handle
            self.__old_handle = None

        def __enter__(self):
            self.__old_handle = glfw.glfwGetCurrentContext()
            if self.__next_handle is not None:
                glfw.glfwMakeContextCurrent(self.__next_handle)
            return self.__old_handle

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                return

            glfw.glfwMakeContextCurrent(self.__old_handle)

    WINDOW_NAME = "Detection Preview"

    def __init__(self, parent: Plugin, path: Path):
        self.path = path
        self.parent = parent
        self.__window = None

    def __bool__(self):
        return self.__window is not None

    def show(self):
        if self.__window is not None:
            raise RuntimeError("Window is already shown.")

        eye0_data = tuple(self.path.glob("eye0_*.png"))
        if len(eye0_data) == 0:
            return

        frame_index = 0

        def on_key(window, key, _scancode, action, _mods):
            nonlocal frame_index

            # Respond only to key press
            if action != glfw.GLFW_PRESS:
                return

            if key == glfw.GLFW_KEY_LEFT and frame_index > 0:
                frame_index -= 1
                PreviewWindow._draw_frame(window, eye0_data, frame_index)
            elif key == glfw.GLFW_KEY_RIGHT and frame_index < len(eye0_data) - 1:
                frame_index += 1
                PreviewWindow._draw_frame(window, eye0_data, frame_index)

        def on_close(_window):
            self.parent.notify_all(
                {"subject": Detection_Preview.NOTIFICATION_PREVIEW_CLOSE}
            )

        first_frame = cv2.imread(str(eye0_data[0]))
        with PreviewWindow.WindowContextManager() as active_window:
            glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, False)
            glfw.glfwWindowHint(glfw.GLFW_ICONIFIED, False)

            self.__window = glfw.glfwCreateWindow(
                first_frame.shape[1],
                first_frame.shape[0],
                PreviewWindow.WINDOW_NAME,
                monitor=None,
                share=active_window,
            )

            # Reset default
            glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, True)
            glfw.glfwWindowHint(glfw.GLFW_ICONIFIED, True)

            glfw.glfwSetKeyCallback(self.__window, on_key)
            glfw.glfwSetWindowCloseCallback(self.__window, on_close)
            glfw.glfwMakeContextCurrent(self.__window)
            basic_gl_setup()
            glfw.glfwSwapInterval(0)

        PreviewWindow._draw_frame(self.__window, eye0_data, 0)

    def close(self):
        if self.__window is None:
            raise RuntimeError("Window is already closed.")

        with PreviewWindow.WindowContextManager():
            glfw.glfwDestroyWindow(self.__window)
            self.__window = None

    @staticmethod
    def _draw_frame(window, files, index):
        file = files[index]
        eye_id, frame_id, confidence = file.stem.split("_")

        frame = cv2.imread(str(file))
        PreviewWindow._draw_text(
            frame,
            "Preview {}/{} ({})".format(index + 1, len(files), eye_id),
            (20, frame.shape[0] - 60),
        )
        PreviewWindow._draw_text(
            frame, "Confidence: {}".format(confidence[-6:]), (20, frame.shape[0] - 30)
        )

        with PreviewWindow.WindowContextManager(window):
            clear_gl_screen()
            make_coord_system_norm_based()
            draw_gl_texture(frame, interpolation=False)
            glfw.glfwSwapBuffers(window)

    @staticmethod
    def _draw_text(frame, string, position):
        cv2.putText(
            frame,
            string,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (157, 233, 68),
            2,
            cv2.LINE_AA,
            False,
        )


class Detection_Preview(Plugin):
    NOTIFICATION_PREVIEW_SHOW = "preview.show"
    NOTIFICATION_PREVIEW_CLOSE = "preview.close"

    icon_chr = "P"
    order = 0.6

    def __init__(
        self,
        g_pool,
        frames_per_frame: int = 120,
        folder: str = "preview",
        should_show: bool = True,
    ):
        super().__init__(g_pool)

        self.frames_per_frame = frames_per_frame
        self.folder = folder
        self.should_show = should_show

        self.__command_sender = None
        self.__worker = None
        self.__status_receiver = None
        self.__generator = None
        self.__window = None

    @property
    def folder(self):
        return self.__folder

    @folder.setter
    def folder(self, folder):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.__folder = folder

    def recent_events(self, _events):
        if self.__status_receiver is not None:
            try:
                if self.__status_receiver.poll():
                    status = self.__status_receiver.recv()
                    if isinstance(status, Exception):
                        raise status
                    else:
                        logger.info("{}".format(status))
            except BrokenPipeError:
                self.__status_receiver = None
                self.__command_sender = None

    def on_notify(self, notification):
        subject = notification["subject"]
        if subject == "recording.started" and self.__worker is None:
            path = self.folder
            if not path.is_absolute() or not path.is_dir():
                recording_path = Path(notification["rec_path"])
                path = recording_path / path
                path.mkdir(parents=True)

            self.__generator = self.__create_generator(path)
            self.__worker = Process(
                target=PreviewGenerator.generate, args=(self.__generator,), daemon=True
            )
            self.__worker.start()

        elif (
            subject == "recording.stopped"
            and self.__worker is not None
            and self.__worker.is_alive()
        ):
            self.__command_sender.send("exit")
            self.__worker.join(3)
            assert self.__worker.exitcode is not None, "Joining failed."

            logger.info("Stopping generation of previews.")
            if len(list(self.__generator.folder.glob("*.png"))) == 0:
                logger.warning(
                    "No previews were generated. Was the Frame Publisher activated?!"
                )
            elif self.should_show:
                self.notify_all(
                    {"subject": Detection_Preview.NOTIFICATION_PREVIEW_SHOW}
                )

            # Reset process properties
            self.__worker = None
            self.__status_receiver = None
            self.__command_sender = None

        elif (
            subject == Detection_Preview.NOTIFICATION_PREVIEW_SHOW
            and self.__generator is not None
            and self.__window is None
        ):
            self.__window = PreviewWindow(self, self.__generator.folder)
            self.__window.show()

        elif (
            subject == Detection_Preview.NOTIFICATION_PREVIEW_CLOSE
            and self.__window is not None
            and bool(self.__window)
        ):
            self.__window.close()
            self.__window = None

    def get_init_dict(self):
        return {
            "frames_per_frame": self.frames_per_frame,
            "folder": str(self.folder),
            "should_show": self.should_show,
        }

    def clone(self):
        return Detection_Preview(**self.get_init_dict())

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Preview of pupil detection"
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
                label="Frame interval",
            )
        )
        self.menu.append(ui.Text_Input("folder", self, label="Storage"))
        self.menu.append(
            ui.Switch("should_show", self, label="Show preview after recording")
        )

    def deinit_ui(self):
        self.remove_menu()

    def __create_generator(self, folder: Path) -> "PreviewGenerator":
        command_receiver, self.__command_sender = Pipe(False)
        self.__status_receiver, status_sender = Pipe(False)
        self.__worker = None
        return PreviewGenerator(
            url=self.g_pool.ipc_sub_url,
            command_pipe=command_receiver,
            exception_pipe=status_sender,
            frame_per_frames=self.frames_per_frame,
            folder=folder,
        )

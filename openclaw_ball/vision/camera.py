class RealSenseCamera:
    """Small adapter; pyrealsense2 is imported only when hardware is requested."""

    def __init__(
        self,
        color_width=1280,
        color_height=720,
        depth_width=848,
        depth_height=480,
        fps=30,
        warmup_frames=60,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is not installed in this Python environment") from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Match the already-validated meter project stream profiles.
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        # D435 auto white balance needs about 2 s in this scene to converge.
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames(timeout_ms=5000)

    def capture(self):
        import numpy as np

        try:
            frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError as exc:
            raise RuntimeError("D435 did not deliver frames within 5 seconds") from exc
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("incomplete RealSense frame")
        # Aligned depth pixels use the color camera geometry.
        intrinsics = color.profile.as_video_stream_profile().intrinsics
        return np.asanyarray(color.get_data()), depth, intrinsics

    def close(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

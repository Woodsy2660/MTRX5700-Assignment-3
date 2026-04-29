# Offline test script - runs on Mac without ROS2
# Reads the .mcap bag, decodes camera + lidar frames, runs detection,
# saves annotated images to aruco_frames/ so we can check what's being detected

import os
import json
import struct
import numpy as np
import cv2
from mcap.reader import make_reader
from landmark_detector import LandmarkDetector

BAG_PATH   = "/Users/lachlandowns/Desktop/bag1/assignment3_0.mcap"
OUT_DIR    = "/Users/lachlandowns/Desktop/aruco_frames"
SAVE_EVERY = 1      # 1 = save every frame, 5 = every 5th etc
MAX_SAVE   = None   # set a number to cap how many frames get saved

os.makedirs(OUT_DIR, exist_ok=True)


# CDR decoding - normally ROS2 handles this automatically but we need to do it
# manually here since we're reading the bag directly without ROS2

def _make_cdr_readers(data: bytes):
    pos = [4]  # skip the 4-byte CDR encapsulation header

    def align4():
        pos[0] = (pos[0] + 3) & ~3

    def read_u32():
        align4()
        v = struct.unpack_from("<I", data, pos[0])[0]
        pos[0] += 4
        return v

    def read_u8():
        v = data[pos[0]]
        pos[0] += 1
        return v

    def read_string():
        length = read_u32()
        s = data[pos[0]:pos[0] + length - 1].decode("utf-8", errors="replace")
        pos[0] += length
        return s

    def current_pos():
        return pos[0]

    return read_u32, read_u8, read_string, current_pos


def decode_image_msg(data: bytes) -> np.ndarray | None:
    try:
        read_u32, read_u8, read_string, cur = _make_cdr_readers(data)

        read_u32(); read_u32()  # stamp.sec, stamp.nanosec
        read_string()           # frame_id
        height   = read_u32()
        width    = read_u32()
        encoding = read_string()
        read_u8()               # is_bigendian
        read_u32()              # step - not needed, we reshape using height/width
        data_length = read_u32()
        pixel_bytes = data[cur():cur() + data_length]

        if len(pixel_bytes) < data_length:
            return None

        arr = np.frombuffer(pixel_bytes, dtype=np.uint8)
        enc = encoding.lower()

        if enc in ("bgra8", "bgra"):
            return cv2.cvtColor(arr.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
        elif enc in ("rgba8", "rgba"):
            return cv2.cvtColor(arr.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
        elif enc == "rgb8":
            return cv2.cvtColor(arr.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            return arr.reshape(height, width, 3)
        elif enc in ("mono8", "8uc1"):
            return cv2.cvtColor(arr.reshape(height, width), cv2.COLOR_GRAY2BGR)
        else:
            print(f"  [!] Unknown encoding '{encoding}'")
            return None

    except Exception as e:
        print(f"  [!] Image decode error: {e}")
        return None


def decode_pointcloud_msg(data: bytes) -> np.ndarray | None:
    # sensor_msgs/PointCloud - each point is 3x float32 (x, y, z), we only need x, y
    try:
        read_u32, _, read_string, cur = _make_cdr_readers(data)

        read_u32(); read_u32()  # stamp
        read_string()           # frame_id

        n_points = read_u32()
        if n_points == 0:
            return np.zeros((0, 2), dtype=np.float32)

        offset = cur()
        raw = np.frombuffer(data[offset:offset + n_points * 12], dtype=np.float32)
        pts = raw.reshape(n_points, 3)
        return pts[:, :2]

    except Exception as e:
        print(f"  [!] PointCloud decode error: {e}")
        return None


def main():
    print(f"\nBag  : {BAG_PATH}")
    print(f"Out  : {OUT_DIR}\n")

    detector    = LandmarkDetector()
    frame_count = 0
    save_count  = 0

    # buffer recent lidar scans so we can match them to camera frames by timestamp
    lidar_buffer: dict[float, np.ndarray] = {}
    MAX_LIDAR_BUFFER = 50

    with open(BAG_PATH, "rb") as f:
        reader = make_reader(f)

        for _, channel, message in reader.iter_messages(
            topics=["/camera/image_raw", "/pointcloud2d"]
        ):
            t = message.log_time / 1e9

            if channel.topic == "/pointcloud2d":
                pts = decode_pointcloud_msg(message.data)
                if pts is not None:
                    lidar_buffer[t] = pts
                    if len(lidar_buffer) > MAX_LIDAR_BUFFER:
                        del lidar_buffer[min(lidar_buffer)]
                continue

            frame_count += 1
            img = decode_image_msg(message.data)
            if img is None:
                continue

            # grab the closest lidar scan to this camera frame
            pointcloud_xy = None
            if lidar_buffer:
                closest_t = min(lidar_buffer, key=lambda lt: abs(lt - t))
                if abs(closest_t - t) < 0.5:
                    pointcloud_xy = lidar_buffer[closest_t]

            observations = detector.process_image(img, t, pointcloud_xy)
            annotated    = detector.annotate(img, observations)

            for obs in observations:
                status = "NEW" if obs.is_new else f"revisit #{detector.registry[obs.tag_id]['count']}"
                r_str  = f"range={obs.range_m:.2f}m" if obs.range_m > 0 else "range=NO LIDAR"
                print(f"  frame={frame_count:4d}  t={t:.3f}s  id={obs.tag_id:3d}  "
                      f"{status:<14}  {r_str}  bearing={np.rad2deg(obs.bearing):+.1f}deg")

            should_save = (frame_count % SAVE_EVERY == 0)
            if MAX_SAVE is not None:
                should_save = should_save and (save_count < MAX_SAVE)

            if should_save:
                ids_str = "_".join(str(o.tag_id) for o in observations) or "none"
                fname   = f"frame_{frame_count:04d}_t{t:.3f}_{ids_str}.jpg"
                cv2.imwrite(os.path.join(OUT_DIR, fname), annotated)
                save_count += 1

    print(f"\n{'='*60}")
    print(f"Frames processed : {frame_count}")
    print(f"Frames saved     : {save_count}")
    print(f"Unique landmarks : {len(detector.registry)}\n")

    for tag_id in sorted(detector.registry):
        e = detector.registry[tag_id]
        print(f"  id={tag_id:3d}  seen {e['count']:3d}x  "
              f"first={e['first_t']:.2f}s  last={e['last_t']:.2f}s")

    registry_path = os.path.join(OUT_DIR, "landmark_registry.json")
    with open(registry_path, "w") as jf:
        json.dump(detector.registry, jf, indent=2)
    print(f"\nRegistry saved to {registry_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

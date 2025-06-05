import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from typing import List, Tuple, Union
import matplotlib.patches as patches
# TruckScenes utils
from truckscenes.utils.geometry_utils import transform_matrix, view_points, BoxVisibility
from truckscenes.utils.data_classes import LidarPointCloud



def _render_pc_sample_data(trucksc,
                               sensor_modality: str,
                               sample_data_token: str,
                               with_anns: bool = True,
                               selected_anntokens: List[str] = None,
                               box_vis_level: BoxVisibility = BoxVisibility.ANY,
                               axes_limit: Union[List[float], Tuple[float], float] = 40,
                               ax: Axes = None,
                               nsweeps: int = 1,
                               use_flat_vehicle_coordinates: bool = True,
                               point_scale: float = 1.0,
                               cmap: str = 'viridis',
                               cnorm: bool = True) -> None:

        intensities = []
        points = []

        for sd_token in sample_data_token:
            sd_record = trucksc.get('sample_data', sd_token)

            sample_rec = trucksc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_LEFT'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = trucksc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                # Get aggregated lidar point cloud in lidar frame.
                pc, _ = LidarPointCloud.from_file_multisweep(trucksc, sample_rec,
                                                             chan, ref_chan,
                                                             nsweeps=nsweeps)
                # === Z Filter: 
                z = pc.points[2, :]  
                z_min = -1.7         
                z_max = 2.0          

                mask = (z >= z_min) & (z <= z_max)
                pc.points = pc.points[:, mask]  

                velocities = None
                intensity = pc.points[3, :]
            

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may
            # not be perfectly upright.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = trucksc.get('calibrated_sensor',
                                             ref_sd_record['calibrated_sensor_token'])
                pose_record = trucksc.get('ego_pose',
                                               ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose
                # (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2),
                               vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

                # Rotate upwards
                vehicle_flat_up_from_vehicle_flat = np.eye(4)
                rotation_axis = Quaternion(matrix=viewpoint[:3, :3])
                vehicle_flat_up_from_vehicle_flat[:3, :3] = \
                    Quaternion(axis=rotation_axis.rotate([0, 0, 1]),
                               angle=np.pi/2).rotation_matrix
                viewpoint = np.dot(vehicle_flat_up_from_vehicle_flat, viewpoint)
            else:
                viewpoint = np.eye(4)

            # Show point cloud
            points.append(view_points(pc.points[:3, :], viewpoint, normalize=False))
            intensities.append(intensity)

        points = np.concatenate(points, axis=1)
        intensities = np.concatenate(intensities, axis=0)

        # Colormapping
        if cnorm:
            norm = Normalize(vmin=np.min(intensities), vmax=np.max(intensities), clip=True)
        else:
            norm = None
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        colors = mapper.to_rgba(intensities)[..., :3]

        point_scale = point_scale * 0.4 if sensor_modality == 'lidar' else point_scale * 3.0
        ax.scatter(points[0, :], points[1, :], marker='o',
                   c=colors, s=point_scale, edgecolors='none')

        # Show ego vehicle
        ax.plot(0, 0, 'x', color='red')

        # Show boxes
        if with_anns:
            # Get boxes in lidar frame.box_vis_level
            _, boxes, _ = trucksc.get_sample_data(
                ref_sd_token, box_vis_level=box_vis_level, selected_anntokens=selected_anntokens,
                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates
            )



            # Render boxes
            for box in boxes:
                c = np.array(trucksc.colormap[box.name]) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=2.0)
                print(box)
        # Limit visible range.
        ax.set_xlim(-axes_limit[0], axes_limit[0])
        ax.set_ylim(-axes_limit[1], axes_limit[1])


def render_sample_data(trucksc,
                           sample_data_token: str,
                           with_anns: bool = True,
                           selected_anntokens: List[str] = None,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: Union[List[float], Tuple[float], float] = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           use_flat_vehicle_coordinates: bool = True,
                           point_scale: float = 1.0,
                           cmap: str = 'viridis',
                           cnorm: bool = True) -> None:

        if not isinstance(sample_data_token, list):
            sample_data_token = [sample_data_token]

        if not isinstance(cmap, Colormap):
            cmap = plt.get_cmap(cmap)

        if not isinstance(axes_limit, (list, tuple)):
            axes_limit = [axes_limit, axes_limit]

        # Determine sensor modality
        sensor_modality = trucksc.get('sample_data', sample_data_token[0])['sensor_modality']

        # Render Point Cloud data
        if sensor_modality in ['lidar', 'radar']:
            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render point cloud data onto axis
            _render_pc_sample_data(trucksc, sensor_modality, sample_data_token, False,
                                        selected_anntokens, box_vis_level, axes_limit,
                                        ax, nsweeps, use_flat_vehicle_coordinates,
                                        point_scale, cmap, cnorm)



        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(sensor_modality.upper(), labels_type=''))
        ax.set_aspect('equal')

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=800)


def get_bev_box(box):
        view = np.eye(4)
        corners = view_points(box.corners(), view, normalize=False)[:2, :]  # Shape (2, 8)

        x_min = np.min(corners[0, :])
        x_max = np.max(corners[0, :])
        y_min = np.min(corners[1, :])
        y_max = np.max(corners[1, :])

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return x_center, y_center, width, height

IMG_WIDTH = 2000  # pixel
IMG_HEIGHT = 2000
AXES_LIMIT = 40  # meter → [-40, +40]  80x80 m

def convert_to_yolo_coords(x, y, w, h):
    x_px = (x + AXES_LIMIT) / (2 * AXES_LIMIT)
    y_px = (y + AXES_LIMIT) / (2 * AXES_LIMIT)
    w_px = w / (2 * AXES_LIMIT)
    h_px = h / (2 * AXES_LIMIT)

    # Clipping tra 0 e 1
    x_px = np.clip(x_px, 0.0, 1.0)
    y_px = np.clip(y_px, 0.0, 1.0)
    w_px = np.clip(w_px, 0.0, 1.0)
    h_px = np.clip(h_px, 0.0, 1.0)

    return x_px, y_px, w_px, h_px


def save_yolo_labels(trucksc, sample_token, out_dir="DATASET/label", class_map=None):
    os.makedirs(out_dir, exist_ok=True)
    sample = trucksc.get("sample", sample_token)

    # Use LIDAR_LEFT as reference channel
    ref_chan = "LIDAR_LEFT"
    ref_sd_token = sample["data"].get(ref_chan)
    if ref_sd_token is None:
        print(f"Nessun {ref_chan} per sample {sample_token}")
        return

    # Get the reference sample data record
    _, boxes, _ = trucksc.get_sample_data(
        ref_sd_token,
        use_flat_vehicle_coordinates=True
    )

    out_path = os.path.join(out_dir, f"{sample_token}.txt")
    with open(out_path, "w") as f:
        for box in boxes:
            x, y, w, h = get_bev_box(box)
            x_norm, y_norm, w_norm, h_norm = convert_to_yolo_coords(x, y, w, h)

            if class_map is None:
                class_id = 0
            else:
                class_id = class_map.get(box.name, 0)

            f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Salvato: {out_path}")


def get_class_map(trucksc):
    all_class_names = set()
    for sample in trucksc.sample:
        for ann_token in sample["anns"]:
            box = trucksc.get_box(ann_token)
            all_class_names.add(box.name)
    sorted_class_names = sorted(all_class_names)
    print(sorted_class_names)
    return {name: idx for idx, name in enumerate(sorted_class_names)}


def render_lidar_fused(trucksc,
                       sample_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       nsweeps: int = 1,
                       out_path: str = None,
                       verbose: bool = True) -> None:

    record = trucksc.get('sample', sample_token)

    lidar_tokens = []
    for channel, token in record['data'].items():
        sd_record = trucksc.get('sample_data', token)
        if sd_record['sensor_modality'] == 'lidar':
            lidar_tokens.append(token)

    if len(lidar_tokens) == 0:
        print(f"Nessun LIDAR per {sample_token}")
        return

    # Graphics setup
    fig_size_inch = 10
    dpi = 200
    fig, ax = plt.subplots(1, 1, figsize=(fig_size_inch, fig_size_inch), dpi=dpi)

    # Rendering of all LIDAR data
    render_sample_data(
        trucksc,
        sample_data_token=lidar_tokens,   
        with_anns=True,
        box_vis_level=box_vis_level,
        axes_limit=(40, 40),  
        ax=ax,
        nsweeps=nsweeps,
        use_flat_vehicle_coordinates=True,
        point_scale=1.0,
        cmap='viridis',
        cnorm=True
    )

    ax.set_title("")
    ax.axis("off")

    out_img_path = f"DATASET/immagine/{sample_token}.png"
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

    plt.savefig(
        out_img_path,
        bbox_inches='tight',   
        pad_inches=0,
        transparent=False
    )
    plt.close(fig)



if __name__ == "__main__":
    try:
        trucksc  
    except NameError:
        raise RuntimeError("Please load the TruckScenes dataset before running this script.")


    output_img_dir = "DATASET/immagine"
    os.makedirs(output_img_dir, exist_ok=True)
    maps=get_class_map(trucksc)

    sample_tokens = [s['token'] for s in trucksc.sample] 

    for i, sample_token in enumerate(sample_tokens):

        print(f"[{i+1}/{len(sample_tokens)}] Rendering sample: {sample_token}")
        try:
            render_lidar_fused(trucksc, sample_token=sample_token, nsweeps=1)
            save_yolo_labels(trucksc, sample_token=sample_token, class_map=maps)
        except Exception as e:
            print(f"Error in rendering of {sample_token}: {e}")
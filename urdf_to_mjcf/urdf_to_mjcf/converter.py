# urdf_to_mjcf/converter.py

import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import trimesh
import numpy as np
import argparse
import logging
from scipy.spatial.transform import Rotation as R


def parse_urdf(urdf_path):
    """Parse the URDF XML file and return the root element.
    Args:        urdf_path: Path to the URDF file.
    Returns:        root: XML root element of the URDF.
    """
    tree = ET.parse(urdf_path)
    return tree.getroot()


def build_urdf_maps(urdf_root):
    """Build maps of links, joints, and their relationships from the URDF XML.
    Returns:
        links: dict of link names to XML elements
        joints: dict of joint names to their type and limits
        children: dict mapping parent link names to lists of child link names
    """
    links, joints, children = {}, {}, {}
    for link in urdf_root.findall("link"):
        links[link.attrib["name"]] = link

    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]

        joint_type = joint.attrib.get("type", "revolute")
        joint_name = joint.attrib["name"]

        limit_elem = joint.find("limit")
        joint_limits = {}
        if limit_elem is not None:
            if "lower" in limit_elem.attrib:
                joint_limits["lower"] = float(limit_elem.attrib["lower"])
            if "upper" in limit_elem.attrib:
                joint_limits["upper"] = float(limit_elem.attrib["upper"])

        joints[child] = {
            "name": joint_name,
            "type": joint_type,
            "element": joint,  # keep full XML for other use
            "limit": joint_limits,
        }

        children.setdefault(parent, []).append(child)

    return links, joints, children


def find_root_link(links, joints):
    """Find the root link in the URDF structure.
    The root link is defined as a link that is not a child of any other link.
    """
    all_links = set(links.keys())
    child_links = set(joints.keys())
    roots = list(all_links - child_links)
    return roots[0] if roots else None


def enforce_valid_inertia(ixx, iyy, izz):
    """
    Ensure the inertia values are valid and non-negative.
    If any value is negative or zero, adjust it to be slightly larger than the sum of the other two.
    """
    ixx, iyy, izz = float(ixx), float(iyy), float(izz)
    eps = 1e-8
    if ixx + iyy < izz:
        izz = ixx + iyy + eps
    if iyy + izz < ixx:
        ixx = iyy + izz + eps
    if izz + ixx < iyy:
        iyy = izz + ixx + eps
    return f"{ixx:.8f} {iyy:.8f} {izz:.8f}"


def compute_inertial_from_stl(stl_path: str, target_mass: float):
    """Compute the center of mass, inertia tensor, and quaternion from an STL file.
    Args:
        stl_path: Path to the STL file.
        target_mass: Desired mass for the object.
    Returns:
        pos_str: Position string in the format "x y z".
        quat_str: Quaternion string in the format "x y z w".
        diaginertia_str: Diagonal inertia tensor string in the format "ixx iyy izz".
    """
    mesh = trimesh.load(stl_path)
    center_of_mass = mesh.center_mass
    inertia_tensor = mesh.moment_inertia
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    scaled_inertia = eigvals * target_mass

    T = np.eye(4)
    T[:3, :3] = eigvecs
    quat = trimesh.transformations.quaternion_from_matrix(T)

    pos_str = f"{center_of_mass[0]:.8f} {center_of_mass[1]:.8f} {center_of_mass[2]:.8f}"
    quat_str = f"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}"
    diaginertia_str = (
        f"{scaled_inertia[0]:.8e} {scaled_inertia[1]:.8e} {scaled_inertia[2]:.8e}"
    )
    return pos_str, quat_str, diaginertia_str


def build_mjcf_body(link_name, links, joints, children):
    """Recursively build the MJCF body element for a given link
    Args:
        link_name: Name of the link to build.
        links: Dictionary of link names to their XML elements.
        joints: Dictionary of joint names to their info.
        children: Dictionary mapping parent link names to lists of child link names.
    Returns:
        body: XML Element representing the MJCF body for the link.
    """
    # Check if the link exists in the URDF links
    if link_name not in links:
        print(f"[ERROR] Link '{link_name}' not found in URDF links.")
        return None

    link = links[link_name]
    body = ET.Element("body", name=link_name)

    # Get joint info and XML element
    joint_info = joints.get(link_name)
    joint_elem = joint_info["element"] if joint_info else None

    # Set body position from joint origin
    if joint_elem is not None:
        origin = joint_elem.find("origin")
        if origin is not None:
            pos = origin.attrib.get("xyz", "0 0 0")
            rpy = origin.attrib.get("rpy", "0 0 0")

            # Convert RPY to quaternion
            roll, pitch, yaw = map(float, rpy.split())
            rot = R.from_euler("xyz", [roll, pitch, yaw])
            quat = rot.as_quat()  # [x, y, z, w]

            # MuJoCo wants [w, x, y, z]
            quat_str = f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}"
            body.set("pos", pos)
            body.set("quat", quat_str)
        else:
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
    else:
        body.set("pos", "0 0 0")
        body.set("quat", "1 0 0 0")

    # -------------------------------
    # Inertial (prefer STL if available)
    # -------------------------------
    inertial = link.find("inertial")
    visual = link.find("visual")
    if inertial is not None:
        origin = inertial.find("origin")
        pos = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        mass = inertial.find("mass").attrib["value"]
        inertia = inertial.find("inertia")
        ixx, iyy, izz = (
            inertia.attrib["ixx"],
            inertia.attrib["iyy"],
            inertia.attrib["izz"],
        )

        mesh_tag = None
        mesh_file = ""
        mesh_path = ""

        if visual is not None:
            mesh_tag = visual.find("geometry/mesh")
            if mesh_tag is not None:
                mesh_file = os.path.basename(mesh_tag.attrib["filename"])
                mesh_path = os.path.join(
                    "/home/hv/robot-model-package-formatter/urdf_to_mjcf/tests",
                    mesh_file,
                )

        if mesh_tag is not None and os.path.exists(mesh_path):
            try:
                pos_str, quat_str, diaginertia_str = compute_inertial_from_stl(
                    mesh_path, float(mass)
                )
                print(f"the diaginertia_str is: {diaginertia_str}")
                diaginertia_vals = [float(v) for v in diaginertia_str.split()]
                mass, diaginertia_str = enforce_minimum_inertial_values(
                    mass, diaginertia_vals
                )

                ET.SubElement(
                    body,
                    "inertial",
                    pos=pos_str,
                    quat=quat_str,
                    mass=mass,
                    diaginertia=diaginertia_str,
                )
            except Exception as e:
                print(f"[ERROR] Inertial computation failed for {link_name}: {e}")
                diaginertia = enforce_valid_inertia(ixx, iyy, izz)
                ET.SubElement(
                    body, "inertial", pos=pos, mass=mass, diaginertia=diaginertia
                )
        else:
            print(f"[WARNING] Mesh not found: {mesh_path} — using URDF inertia")
            diaginertia = enforce_valid_inertia(ixx, iyy, izz)
            ET.SubElement(body, "inertial", pos=pos, mass=mass, diaginertia=diaginertia)

    # -------------------------------
    # Joint
    # -------------------------------
    if joint_elem is not None:
        axis_tag = joint_elem.find("axis")
        axis = axis_tag.attrib.get("xyz", "0 0 1") if axis_tag is not None else "0 0 1"

        joint_pos = "0 0 0"  # MuJoCo assumes it's at the child frame origin

        joint_type = joint_elem.attrib["type"]
        if joint_type == "revolute" or joint_type == "continuous":
            mjcf_type = "hinge"
        elif joint_type == "prismatic":
            mjcf_type = "slide"
        elif joint_type == "fixed":
            mjcf_type = None  # No joint needed
        else:
            print(f"[WARNING] Unsupported joint type: {joint_type}")
            mjcf_type = None

        if mjcf_type:
            joint_attribs = {
                "name": joint_elem.attrib["name"],
                "type": mjcf_type,
                "pos": joint_pos,
                "axis": axis,
            }

            limit = joint_elem.find("limit")
            if (
                limit is not None
                and "lower" in limit.attrib
                and "upper" in limit.attrib
            ):
                joint_attribs["limited"] = "true"
                joint_attribs["range"] = (
                    f"{limit.attrib['lower']} {limit.attrib['upper']}"
                )
                joint_attribs["damping"] = "100"  # Default damping value

            body.append(ET.Element("joint", joint_attribs))

    # -------------------------------
    # Geom from visual mesh
    # -------------------------------
    if visual is not None:
        mesh_tag = visual.find("geometry/mesh")
        if mesh_tag is not None:
            mesh_file = os.path.basename(mesh_tag.attrib["filename"])
            mesh_name = os.path.splitext(mesh_file)[0]

            origin = visual.find("origin")
            pos = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"

            material = visual.find("material")
            color_tag = material.find("color") if material is not None else None
            rgba = (
                color_tag.attrib.get("rgba", "0.7 0.7 0.7 1")
                if color_tag is not None
                else "0.7 0.7 0.7 1"
            )

            ET.SubElement(body, "geom", type="mesh", mesh=mesh_name, pos=pos, rgba=rgba)

    # -------------------------------
    # Recursively add child bodies
    # -------------------------------
    for child in children.get(link_name, []):
        child_body = build_mjcf_body(child, links, joints, children)
        if child_body is not None:
            body.append(child_body)
        else:
            print(f"[WARNING] Skipping child link: {child} (build failed)")

    return body


def enforce_minimum_inertial_values(mass, diaginertia, min_mass=1e-5, min_inertia=1e-8):
    """Ensure mass and inertia values are above minimum thresholds.
    Args:
        mass: The mass value to check.
        diaginertia: List of inertia values [ixx, iyy, izz].
        min_mass: Minimum allowed mass.
        min_inertia: Minimum allowed inertia value for each component.
    Returns:
        Tuple of (mass, diaginertia_str) where diaginertia_str is a space-separated string of inertia values.
    """
    # Ensure mass is float
    mass = float(mass)
    mass = max(mass, min_mass)

    # Ensure each inertia component is float
    diaginertia = [max(float(i), min_inertia) for i in diaginertia]
    diaginertia_str = " ".join(f"{i:.8e}" for i in diaginertia)
    print(f"the remnapped diaginertia is: {diaginertia}")
    return f"{mass:.8f}", diaginertia_str


def add_actuators_auto(mjcf_root, joints):
    """Automatically add actuators for all actuated joints in the MJCF model.
    Args:
        mjcf_root: The root element of the MJCF XML.
        joints: Dictionary of joint names to their info.
    """
    print("[INFO] Adding actuators for all actuated joints...")
    # Create actuator section
    print(f"the joints are: {joints}")
    actuator = ET.SubElement(mjcf_root, "actuator")

    for joint_info in joints.values():
        joint_type = joint_info.get("type", "hinge")
        joint_name = joint_info.get("name")  # Actual joint name

        # # Only handle actuated joint types
        if joint_type not in ("hinge", "slide", "revolute", "prismatic"):
            continue

        # Get control range from joint limits
        limit = joint_info.get("limit", {})
        lower = limit.get("lower")
        upper = limit.get("upper")
        print(
            f"Adding actuator for joint: {joint_name}, type: {joint_type}, limits: {lower}, {upper}"
        )
        ET.SubElement(
            actuator,
            "position",
            name=joint_name,
            joint=joint_name,
            ctrllimited="true",
            ctrlrange=f"{lower} {upper}",
            kp="1000",
            forcelimited="true",
            forcerange="-1000 1000",
            user="1",
        )


def convert_urdf_to_mjcf(urdf_path: str, output_path: str) -> None:
    """Convert a URDF file to MJCF format.
    Args:
        urdf_path: Path to the input URDF file.
        output_path: Path to save the output MJCF file.
    Raises:
        RuntimeError: If the URDF cannot be parsed or if the root link cannot be determined.
    """
    urdf_root = parse_urdf(urdf_path)
    links, joints, children = build_urdf_maps(urdf_root)
    root_link = find_root_link(links, joints)

    if not root_link:
        raise RuntimeError("Could not determine root link of URDF")

    mjcf = ET.Element("mujoco", model=os.path.splitext(os.path.basename(urdf_path))[0])
    ET.SubElement(mjcf, "compiler", angle="radian")
    ET.SubElement(mjcf, "option", gravity="0 0 -9.81")

    # Asset section for meshes
    asset = ET.SubElement(mjcf, "asset")
    mesh_names = set()
    for link in links.values():
        visual = link.find("visual")
        if visual is not None:
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                mesh_file = os.path.basename(mesh.attrib["filename"])
                mesh_name = os.path.splitext(mesh_file)[0]
                if mesh_name not in mesh_names:
                    ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file)
                    mesh_names.add(mesh_name)

    # Main worldbody
    worldbody = ET.SubElement(mjcf, "worldbody")
    body_elem = build_mjcf_body(root_link, links, joints, children)
    if body_elem is not None:
        worldbody.append(body_elem)
    else:
        raise RuntimeError(f"Failed to build body for root link: {root_link}")
    add_actuators_auto(mjcf, joints)

    # Pretty-print the XML
    rough_string = ET.tostring(mjcf, encoding="utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml)

    logging.info(f"MJCF written to: {output_path}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Convert URDF to MJCF (MuJoCo XML)")
    parser.add_argument("urdf", type=str, help="Path to input URDF file")
    parser.add_argument("output", type=str, help="Path to output MJCF file")
    args = parser.parse_args()
    convert_urdf_to_mjcf(args.urdf, args.output)


if __name__ == "__main__":
    main()

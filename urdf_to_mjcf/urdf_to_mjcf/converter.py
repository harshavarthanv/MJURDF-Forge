# urdf_to_mjcf/converter.py

import os
import xml.etree.ElementTree as ET
import xml.dom.minidom


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    return tree.getroot()


def build_urdf_maps(urdf_root):
    links, joints, children = {}, {}, {}
    for link in urdf_root.findall("link"):
        links[link.attrib["name"]] = link

    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        joints[child] = joint
        children.setdefault(parent, []).append(child)

    return links, joints, children


def find_root_link(links, joints):
    all_links = set(links.keys())
    child_links = set(joints.keys())
    roots = list(all_links - child_links)
    return roots[0] if roots else None


def enforce_valid_inertia(ixx, iyy, izz):
    ixx, iyy, izz = float(ixx), float(iyy), float(izz)
    eps = 1e-8
    if ixx + iyy < izz:
        izz = ixx + iyy + eps
    if iyy + izz < ixx:
        ixx = iyy + izz + eps
    if izz + ixx < iyy:
        iyy = izz + ixx + eps
    return f"{ixx:.8f} {iyy:.8f} {izz:.8f}"


def build_mjcf_body(link_name, links, joints, children):
    link = links[link_name]
    body = ET.Element("body", name=link_name)

    # Set body.pos from joint origin (transform from parent)
    joint = joints.get(link_name)
    if joint is not None:
        origin = joint.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            body.set("pos", origin.attrib["xyz"])
        else:
            body.set("pos", "0 0 0")
    else:
        body.set("pos", "0 0 0")

    # Inertial
    inertial = link.find("inertial")
    if inertial is not None:
        origin = inertial.find("origin")
        pos = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        mass = inertial.find("mass").attrib["value"]
        inertia = inertial.find("inertia")
        ixx, iyy, izz = (
            inertia.attrib.get("ixx", "0"),
            inertia.attrib.get("iyy", "0"),
            inertia.attrib.get("izz", "0"),
        )
        diaginertia = enforce_valid_inertia(ixx, iyy, izz)
        ET.SubElement(body, "inertial", pos=pos, mass=mass, diaginertia=diaginertia)

    # Visual → geom
    visual = link.find("visual")
    if visual is not None:
        mesh_tag = visual.find("geometry/mesh")
        if mesh_tag is not None:
            mesh_file = os.path.basename(mesh_tag.attrib["filename"])
            mesh_name = os.path.splitext(mesh_file)[0]
            origin = visual.find("origin")
            pos = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rgba_tag = visual.find("material/color")
            rgba = rgba_tag.attrib["rgba"] if rgba_tag is not None else "0.7 0.7 0.7 1"
            ET.SubElement(body, "geom", type="mesh", mesh=mesh_name, pos=pos, rgba=rgba)

    # Joint
    if joint is not None:
        axis_tag = joint.find("axis")
        axis = axis_tag.attrib.get("xyz", "0 0 1") if axis_tag is not None else "0 0 1"

        joint_origin = joint.find("origin")
        joint_pos = "0 0 0"  # Local joint position in the child frame
        joint_type = joint.attrib["type"]
        mjcf_type = (
            "hinge"
            if joint_type in ["revolute", "continuous"]
            else "slide" if joint_type == "prismatic" else "free"
        )
        limit = joint.find("limit")
        lower = limit.attrib.get("lower", "0") if limit is not None else "0"
        upper = limit.attrib.get("upper", "0") if limit is not None else "0"

        ET.SubElement(
            body,
            "joint",
            name=joint.attrib["name"],
            type=mjcf_type,
            pos=joint_pos,
            axis=axis,
            limited="true",
            range=f"{lower} {upper}",
        )

    # Recursively add children
    for child in children.get(link_name, []):
        body.append(build_mjcf_body(child, links, joints, children))

    return body


def convert_urdf_to_mjcf(urdf_path: str, output_path: str) -> None:
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
    worldbody.append(build_mjcf_body(root_link, links, joints, children))

    # Pretty-print the XML
    rough_string = ET.tostring(mjcf, encoding="utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml)

    print(f"MJCF written to: {output_path}")

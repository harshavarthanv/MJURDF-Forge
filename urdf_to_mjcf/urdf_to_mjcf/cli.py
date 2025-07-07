# urdf_to_mjcf/cli.py

import argparse
from urdf_to_mjcf.converter import convert_urdf_to_mjcf


def main():
    parser = argparse.ArgumentParser(
        description="Convert URDF to MuJoCo MJCF XML format"
    )
    parser.add_argument("input", help="Path to input URDF file")
    parser.add_argument("output", help="Path to output MJCF XML file")

    args = parser.parse_args()
    convert_urdf_to_mjcf(args.input, args.output)


if __name__ == "__main__":
    main()

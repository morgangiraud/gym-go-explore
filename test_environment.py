import sys

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    required_major = 3
    required_min_minor = 6

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}.{} at least.\nFound: Python {}\n\nMaybe you forgot to activate your conda env?\n".format(
                required_major, required_min_minor, sys.version))
    elif system_minor < required_min_minor:
        raise TypeError(
            "This project requires Python {}.{} at least.\nFound: Python {}\n\nMaybe you forgot to activate your conda env?\n".format(
                required_major, required_min_minor, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()

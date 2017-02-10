#!/bin/bash
set -eo pipefail

version=$(git describe --tags)

confirm() {
    read -r -p "${1:-Are you sure? [y/N]} " response
    case $response in
        [yY][eE][sS]|[yY])
            true
            ;;
        *)
            false
            ;;
    esac
}

echo "Releasing metapy-${version}..."
confirm || exit 1

echo "Creating source distribution..."
python setup.py sdist --formats=gztar

echo "Fetching wheels from GitHub release..."
python get-release.py ${version}

echo "Uploading to PyPI..."
twine upload -s dist/*.{gz,whl}

echo "Done!"

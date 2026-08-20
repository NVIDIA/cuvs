#!/bin/bash
set -e
#set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"
python3 -m venv "$SCRIPT_DIR"/venv
source "$SCRIPT_DIR"/venv/bin/activate
pip install -r "$SCRIPT_DIR"/requirements.txt
VERSION=$("$SCRIPT_DIR"/venv/bin/python3 "$SCRIPT_DIR"/main.py version)
if [[ "$VERSION" =~ ^[0-9.]+$ ]]; then
echo  '****************************************************************'
echo  '*        ___  ________ _     _   _ _   _ _____                 *'
echo  '*        |  \/  |_   _| |   | | | | | | /  ___|                *'
echo  '*        | .  . | | | | |   | | | | | | \ `--.                 *'
echo  '*        | |\/| | | | | |   | | | | | | |`--. \                *'
echo  '*        | |  | |_| |_| |___\ \_/ / |_| /\__/ /                *'
echo  '*        \_|  |_/\___/\_____/\___/ \___/\____/                 *'
echo  '*                                                              *'
echo  '*                                                              *'
echo  '*  ______  _____  _____ _____ _____   ___  ___  _________      *'
echo  '*  | ___ \|  _  ||  _  |_   _/  __ \ / _ \ |  \/  || ___ \     *'
echo  '*  | |_/ /| | | || | | | | | | /  \// /_\ \| .  . || |_/ /     *'
echo  '*  | ___ \| | | || | | | | | | |    |  _  || |\/| ||  __/      *'
echo  '*  | |_/ /\ \_/ /\ \_/ / | | | \__/\| | | || |  | || |         *'
echo  '*  \____/  \___/  \___/  \_/  \____/\_| |_/\_|  |_/\_|         *'
echo  '*                                                              *'
echo  '****************************************************************'
echo Version $VERSION installed successfully
else
  echo $VERSION
fi
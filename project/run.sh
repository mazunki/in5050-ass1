#!/bin/sh
set -eu

PROJECT_USER=in5050-g01
PROJECT_ROOT="/home/${PROJECT_USER}/in5050-ass1/project"
BUILD_DIR="${PROJECT_ROOT}/build"
WORKDIR="${PROJECT_ROOT}/workdir"
REPORT_FILE="report.nsys-rep"

BUILDER="${PROJECT_USER}@in5050"
RUNNER="${PROJECT_USER}@in5050-2016-10"

cd "$(dirname "$0")"

builder() {
	echo "[BUILDER] $*"
	ssh "$BUILDER" "$*"
}

runner() {
	echo "[RUNNER] $*"
	ssh "$RUNNER" "$*"
}

pipeline() {
	echo "[PIPELINE] Updating build server..."
	(set -x; rsync -av --progress . "${BUILDER}:${PROJECT_ROOT}/")

	echo "[PIPELINE] updating cmake..."
	builder "cd '${PROJECT_ROOT}' && rm -rf build && cmake -B build -DCMAKE_TOOLCHAIN_FILE=in5050-toolchain.cmake"

	echo "[PIPELINE] building project..."
	builder "cd '${BUILD_DIR}' && make"

	echo "[PIPELINE] syncing build machine with gpu machine..."
	# (set -x; rsync -av --progress "${BUILDER}:${BUILD_DIR}/" "$RUNNER:${BUILD_DIR}/")
	(set -x; ssh "${BUILDER}" "rsync -av --progress '${BUILD_DIR}/' '${RUNNER}:${BUILD_DIR}/'")
	

	echo "[PIPELINE] running profiling on gpu machine..."
	cmd="${BUILD_DIR}/c63enc -h 288 -w 352 -o output -f 128 /home/in5050-g01/assets/foreman.yuv"
	runner "rm -rf '${WORKDIR}' && mkdir '${WORKDIR}'"
	runner "cd '${WORKDIR}' && nsys profile -o '${REPORT_FILE}' -- ${cmd}"

	echo "[PIPELINE] Fetching profiling report..."
	(set -x; rsync -av --progress "$RUNNER:$WORKDIR/" "workdir/")
}

pipeline


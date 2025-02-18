#!/bin/sh
set -e

PROJECT_ROOT="/home/in5050-g01/in5050-ass1/project"
BUILD_DIR="${PROJECT_ROOT}/build"
WORKDIR="${PROJECT_ROOT}/workdir"
REMOTE_BUILD="in5050:${BUILD_DIR}"
REMOTE_WORKDIR="in5050-gpu:${WORKDIR}"
REPORT_FILE="report.nsys-rep"

BUILDER="in5050"
RUNNER="in5050-gpu"

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
	rsync -av --progress . "${BUILDER}:${PROJECT_ROOT}/"

	# echo "[PIPELINE] updating cmake..."
	# builder "cd '${PROJECT_ROOT}' && rm -rf build && cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake"

	echo "[PIPELINE] building project..."
	builder "cd '${BUILD_DIR}' && make"

	echo "[PIPELINE] syncing build machine with gpu machine..."
	rsync -av --progress "${BUILDER}:${BUILD_DIR}/" "$RUNNER:${BUILD_DIR}/"

	echo "[PIPELINE] running profiling on gpu machine..."
	cmd="${BUILD_DIR}/c63enc -h 288 -w 352 -o output -f 128 /home/in5050-g01/assets/foreman.yuv"
	runner "cd '${WORKDIR}' && nsys profile -o '${REPORT_FILE}' -- ${cmd}"

	echo "[PIPELINE] Fetching profiling report..."
	mkdir -p workdir
	rsync -av --progress "$RUNNER:$WORKDIR/$REPORT_FILE" "workdir/${REPORT_FILE}"
}

pipeline


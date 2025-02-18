#!/bin/sh
set -e

commit() {
	echo commiting
	git add src/ include/ CMakeLists.txt install.sh
	git commit -m 'test commit'
	git push
}

rebuild() {
	echo rebuilding
	cd /home/in5050-g01/in5050-ass1/project/build
	git pull
	make
}

deploy() {
	echo deploying
	scp /home/in5050-g01/in5050-ass1/project/build/c63* in5050-2016-10:/home/in5050-g01/in5050-ass1/project/build/
	
}

profile() {
	echo profiling

	workdir=/home/in5050-g01/in5050-ass1/project/workdir
	mkdir -p "${workdir}"
	cd "${workdir}"
	nsys profile -o report.nsys-rep -- ../build/c63enc -h 288 -w 352 -o output -f 128 /home/in5050-g01/assets/foreman.yuv
	cd
	# rm -r "${workdir}"
}

newreport() {
	echo new report
	echo remote in5050 rebuild deploy
	ssh in5050 /home/in5050-g01/in5050-ass1/project/install.sh rebuild deploy

	echo remote in5050-gpu profile
	ssh in5050-gpu /home/in5050-g01/in5050-ass1/project/install.sh profile

	echo getting report
	scp in5050-gpu:/home/in5050-g01/in5050-ass1/project/workdir/report.nsys-rep .
}

for target; do
	case "${target}" in
		rebuild) rebuild ;;
		deploy) deploy ;;
		profile) profile ;;
		commit) commit ;;
		newreport) newreport ;;
		all) commit && newreport ;;
		*) exit 1 ;;
	esac
done



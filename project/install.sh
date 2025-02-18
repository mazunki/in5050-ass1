#!/bin/sh
set -e

commit() {
	git add src/ include/ CMakeLists.txt install.sh
	git commit -m 'test commit'
	git push
}

rebuild() {
	cd /home/in5050-g01/in5050-ass1/project/build
	git pull
	make
}

deploy() {
	cd /home/in5050-g01/in5050-ass1/project/workdir
	scp c63* in5050-2016-10:ass1/build/
}

profile() {
	cd /home/in5050-g01/in5050-ass1/project/workdir
	nsys profile -o report.nsys-rep -- ../build/c63enc -h 288 -w 352 -o output -f 128 /home/in5050-g01/assets/foreman.yuv
}

newreport() {
	ssh in5050 sh -c 'cd /home/in5050-g01/ass1/project && ./install.sh rebuild deploy'
	ssh in5050-gpu sh -c 'cd /home/in5050-g01/ass1/project && ./install.sh profile'
	scp in5050-gpu:/home/in5050-g01/ass1/workdir/report.nsys-rep .
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



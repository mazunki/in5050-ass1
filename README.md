# IN5050 - Home exam 1 - Code Delivery of group 01

We are presenting two folders since we didn't have a chance to test the last solution. We know it works until commit 0f0775be, but while migrating changes from our legacy/testing repo the server stopped working and we couldn't finish all the migration.

We know the `legacy/` folder works, but it's sort of messy, and is not really our main submission.

Our submission is the `neo/` folder, which can also be found on [github:mazunki/in5050-ass1](https://github.com/mazunki/in5050-ass1). Inside the `project/` subdirectory, there's a README explaining how to run the code automatically on the machine. We mainly test our code with `./run.sh`, which does all the deployment for us.

If we had a bit more time, we would try implementing these ideas, which would probably improve performance minimally.
- Replace shared memory with warp reduction (using `__shfl_*` functions)
- Use persistent kernels to avoid the overhead of creating these per-frame
- Look into using `cudaMemPrefetchAsync` instead of writing with `cudaMemcpyAsync` each frame.
- Consider increasing the framebuffer size (since it seems like we didn't fill up the device memory)
- Replace some of the stream barriers to start async memcopies with events to triggere these functions, since currently the stream synchronization blocks execution.


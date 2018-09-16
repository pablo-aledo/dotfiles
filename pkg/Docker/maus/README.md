Docker MAUS
===========

[View on Docker Hub](https://hub.docker.com/r/stevecassidy/maus/)

A docker container for MAUS (Munich AUtomatic Segmentation) forced alignment
system.  This container makes it easier to run MAUS on non-linux platforms such
as macOS and (probably) Windows.  It contains all of the required dependencies
and setup to run MAUS over audio files.  All options for the original scripts
are available.

This is a command line container, it has no fixed entry point so you must
give a command line to the 'docker run' command.

MAUS is installed in /home/maus.  The following command will run MAUS over the
test file included in the distribution.

```bash
docker run -v `pwd`:/export stevecassidy/maus \
    /home/maus/maus OUT=/export/test.TextGrid \
    OUTFORMAT=TextGrid \
    SIGNAL=/home/maus/test.wav \
    BPF=/home/maus/test.par
```
The -v option maps the current working directory on the host machine (your laptop)
to /export in the container. We write the resulting test.TextGrid file out to
that directory so it should appear in the current directory.

(Note that you don't need to install anything other than Docker to run this. The
first time you run the command above, it will download the container; on subsequent
runs it uses the version it has downloaded already.)

To run MAUS over your own files they need to be inside the directory mapped to
/export in the container. For example:

```bash
docker run -v `pwd`:/export  stevecassidy/maus \
    /home/maus/maus OUT=/export/test.TextGrid \
    OUTFORMAT=TextGrid \
    SIGNAL=/export/test/1_1119_2_22_001-ch6-speaker16.wav \
    BPF=/export/test/1_1119_2_22_001.bpf \
    LANGUAGE=aus
```

All of the other scripts in MAUS should work too (maus.corpus, maus.iter etc).

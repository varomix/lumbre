// TAA on a frame where instances moved: taa_fs.slang, reprojecting surfaces
// by the motion pass's velocity. See the note there on why this is a pipeline
// of its own.

#define TAA_VELOCITY 1
#include "taa_fs.slang"

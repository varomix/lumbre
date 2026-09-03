// No colour output: the shadow pass writes depth only. SDL_GPU still wants a
// fragment shader bound, so this is the smallest legal one.

[shader("fragment")]
void fragmentMain() {}

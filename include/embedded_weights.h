#pragma once

#define INCBIN_PREFIX
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin/incbin.h>

namespace embed{

INCBIN(weights_file, EVALFILE);

}
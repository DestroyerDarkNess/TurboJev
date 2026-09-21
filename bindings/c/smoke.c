#include "include/turbojev.h"
#include <stddef.h>
int main(void) {
  tj_engine *engine = NULL;
  (void) engine;
  return tj_abi_version() == 0;
}

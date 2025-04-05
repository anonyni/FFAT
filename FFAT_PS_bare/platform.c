#include "platform.h"
#include "xil_printf.h"

void init_platform() {
    xil_printf("Platform initialized.\\n");
}

void cleanup_platform() {
    xil_printf("Platform cleanup complete.\\n");
}

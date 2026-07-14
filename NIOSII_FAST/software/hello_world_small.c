/* 
 * "Small Hello World" example. 
 * 
 * This example prints 'Hello from Nios II' to the STDOUT stream. It runs on
 * the Nios II 'standard', 'full_featured', 'fast', and 'low_cost' example 
 * designs. It requires a STDOUT  device in your system's hardware. 
 *
 * The purpose of this example is to demonstrate the smallest possible Hello 
 * World application, using the Nios II HAL library.  The memory footprint
 * of this hosted application is ~332 bytes by default using the standard 
 * reference design.  For a more fully featured Hello World application
 * example, see the example titled "Hello World".
 *
 * The memory footprint of this example has been reduced by making the
 * following changes to the normal "Hello World" example.
 * Check in the Nios II Software Developers Manual for a more complete 
 * description.
 * 
 * In the SW Application project (small_hello_world):
 *
 *  - In the C/C++ Build page
 * 
 *    - Set the Optimization Level to -Os
 * 
 * In System Library project (small_hello_world_syslib):
 *  - In the C/C++ Build page
 * 
 *    - Set the Optimization Level to -Os
 * 
 *    - Define the preprocessor option ALT_NO_INSTRUCTION_EMULATION 
 *      This removes software exception handling, which means that you cannot 
 *      run code compiled for Nios II cpu with a hardware multiplier on a core 
 *      without a the multiply unit. Check the Nios II Software Developers 
 *      Manual for more details.
 *
 *  - In the System Library page:
 *    - Set Periodic system timer and Timestamp timer to none
 *      This prevents the automatic inclusion of the timer driver.
 *
 *    - Set Max file descriptors to 4
 *      This reduces the size of the file handle pool.
 *
 *    - Check Main function does not exit
 *    - Uncheck Clean exit (flush buffers)
 *      This removes the unneeded call to exit when main returns, since it
 *      won't.
 *
 *    - Check Don't use C++
 *      This builds without the C++ support code.
 *
 *    - Check Small C library
 *      This uses a reduced functionality C library, which lacks  
 *      support for buffering, file IO, floating point and getch(), etc. 
 *      Check the Nios II Software Developers Manual for a complete list.
 *
 *    - Check Reduced device drivers
 *      This uses reduced functionality drivers if they're available. For the
 *      standard design this means you get polled UART and JTAG UART drivers,
 *      no support for the LCD driver and you lose the ability to program 
 *      CFI compliant flash devices.
 *
 *    - Check Access device drivers directly
 *      This bypasses the device file system to access device drivers directly.
 *      This eliminates the space required for the device file system services.
 *      It also provides a HAL version of libc services that access the drivers
 *      directly, further reducing space. Only a limited number of libc
 *      functions are available in this configuration.
 *
 *    - Use ALT versions of stdio routines:
 *
 *           Function                  Description
 *        ===============  =====================================
 *        alt_printf       Only supports %s, %x, and %c ( < 1 Kbyte)
 *        alt_putstr       Smaller overhead than puts with direct drivers
 *                         Note this function doesn't add a newline.
 *        alt_putchar      Smaller overhead than putchar with direct drivers
 *        alt_getchar      Smaller overhead than getchar with direct drivers
 *
 */

#include <stdio.h>
#include "sys/alt_stdio.h"
#include "system.h"
#include "altera_avalon_performance_counter.h"
#include "model_weights.h"
#include "neural_network.h"
// Custom function to print floats without modifying the "Small" template properties
void print_small_float(float f, int precision) {
    if (f < 0) {
        printf("-");
        f = -f;
    }

    // Print the integer part
    int integer_part = (int)f;
    printf("%d.", integer_part);

    // Extract and print the fractional part based on requested precision
    float fraction = f - (float)integer_part;
    while (precision > 0) {
        fraction *= 10;
        int digit = (int)fraction;
        printf("%d", digit);
        fraction -= digit;
        precision--;
    }
}
float l0[16], l1[16];  // outputs buffers for each layer

int main() {
	printf("input,output,cycles taken \n");
	float input=-PI;
    for(int i=0 ;i<628 ;i++){
    // 1. Wipe the counter history and turn it on
    	PERF_RESET(PERFORMANCE_COUNTER_0_BASE);
   	    PERF_START_MEASURING(PERFORMANCE_COUNTER_0_BASE);

    	    // 2. Start the stopwatch for Section 1
    	    PERF_BEGIN(PERFORMANCE_COUNTER_0_BASE, 1);

    	    float inp=standardize_input(input,INPUT_MEAN,INPUT_SCALE);
    	           forward_pass(&inp, 1, W0 ,B0, l0, 4);
    	           tanh_activation(l0, 4);

    	           // uart_write("layer2\n\r",9);
    	           forward_pass(l0, 4, W1, B1, l1, 8);
    	           tanh_activation(l1, 8);

    	           // uart_write("layer3\n\r",9);
    	           forward_pass(l1, 8, W2, B2, l0, 8);
    	           tanh_activation(l0, 8);

    	           // uart_write("layer4\n\r",9);
    	           forward_pass(l0, 8, W3, B3, l1, 4);
    	           tanh_activation(l1, 4);

    	           // Layer 4 (final output)
    	           forward_pass(l1, 4, W4, B4, l0, 1);


    	    // 3. Stop the stopwatch for Section 1
    	    PERF_END(PERFORMANCE_COUNTER_0_BASE, 1);

    	    // 4. Extract the hardware register value
    	    unsigned int total_cycles = (unsigned int)perf_get_section_time(PERFORMANCE_COUNTER_0_BASE, 1);
    	    input+=0.01f;
			// 5. Output results using template-safe prints
			print_small_float(input, 4);
			printf(" , ");
			print_small_float(l0[0], 4); // Prints the float down to 4 decimal places
			printf(" , ");
			printf("%u\n", total_cycles);
    }

while(1);
    return 0;
}



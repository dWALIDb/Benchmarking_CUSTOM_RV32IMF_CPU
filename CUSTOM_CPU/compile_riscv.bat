riscv-none-elf-gcc -T newlinker.ld  -Wl,-Map=first.map -fdata-sections -ffunction-sections -Wl,--gc-sections -fno-unsafe-math-optimizations -ffp-contract=off -fno-fast-math -I%CD%/utils/ -o first.elf *.c -O1 -mabi=ilp32f -march=rv32imf -nostartfiles 
riscv-none-elf-objcopy -O binary first.elf first.bin 
riscv-none-elf-objdump -d first.elf > first.asm
flipper.exe first.bin 16384
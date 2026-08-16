CC       ?= gcc
LD       ?= ld
OBJCOPY  ?= objcopy
PYTHON   ?= python3
BUILD    := build
ISO_DIR  := $(BUILD)/iso
ULTRA_ISO_DIR := $(BUILD)/ultra-iso
HF_DIR   := $(BUILD)/hf
MODEL_ID := BananaMind/BananaMind-2-Nano
NANO_REV := c8564d1bd3f6177221ed7e4f63ae5f281a677a1c
HF_BASE  := https://huggingface.co/$(MODEL_ID)/resolve/$(NANO_REV)
HF_CHAT_DIR  := $(BUILD)/hf-chat
CHAT_MODEL_ID := BananaMind/BananaMind-2-Nano-Chat
CHAT_REV := 6def7ab4d1bd8449d9b75b62bfb174f73d3a8ad6
HF_CHAT_BASE := https://huggingface.co/$(CHAT_MODEL_ID)/resolve/$(CHAT_REV)
HF_MICRO2_DIR := $(BUILD)/hf-micro2
MICRO2_BASE := https://huggingface.co/BananaMind/BananaMind-2-Micro/resolve/dbc892b2df93f45d120b07c64673f00e3abb1593
HF_MICROV1_DIR := $(BUILD)/hf-microv1
MICROV1_BASE := https://huggingface.co/BananaMind/MicroBananaMind-v1/resolve/2dd299a219ffabc05bfcd39862e9293c68d3ef92
HF_MINI_DIR := $(BUILD)/hf-mini
MINI_BASE := https://huggingface.co/BananaMind/BananaMind-2-Mini/resolve/6400d0a6dcbe52f3c725291cb97c079c591c05a4

CFLAGS := -m32 -march=i486 -mtune=i486 -mfpmath=387 -m80387 \
	-ffreestanding -fno-pie -fno-stack-protector -fno-builtin \
	-fno-asynchronous-unwind-tables -fno-unwind-tables -nostdlib \
	-Os -Wall -Wextra -Werror -std=c11 -Iinclude
LDFLAGS := -m elf_i386 -T linker.ld -nostdlib

.PHONY: all iso ultra kernel models models-extra download download-extra run run-2 run-4 run-8 \
	run-micro run-microv1 run-mini run-ultra host-chat host-base host-micro host-microv1 host-mini clean check

all: iso

kernel: $(BUILD)/kernel.elf

$(BUILD)/boot.o: boot/boot.S | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/kernel.o: src/kernel.c src/font8x8.h include/bm2n.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/kernel.elf: $(BUILD)/boot.o $(BUILD)/kernel.o linker.ld
	$(LD) $(LDFLAGS) -o $@ $(BUILD)/boot.o $(BUILD)/kernel.o

$(BUILD):
	mkdir -p $@

$(HF_DIR):
	mkdir -p $@

$(HF_CHAT_DIR):
	mkdir -p $@

$(HF_MICRO2_DIR) $(HF_MICROV1_DIR) $(HF_MINI_DIR):
	mkdir -p $@

download: $(HF_DIR)/model.safetensors $(HF_DIR)/tokenizer.json $(HF_DIR)/config.json \
	$(HF_CHAT_DIR)/model.safetensors $(HF_CHAT_DIR)/tokenizer.json $(HF_CHAT_DIR)/config.json \
	download-extra

download-extra: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json \
	$(HF_MICROV1_DIR)/model.safetensors $(HF_MICROV1_DIR)/tokenizer.json $(HF_MICROV1_DIR)/config.json \
	$(HF_MINI_DIR)/model.safetensors $(HF_MINI_DIR)/tokenizer.json $(HF_MINI_DIR)/config.json

$(HF_DIR)/model.safetensors: | $(HF_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_BASE)/model.safetensors

$(HF_DIR)/tokenizer.json: | $(HF_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_BASE)/tokenizer.json

$(HF_DIR)/config.json: | $(HF_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_BASE)/config.json

$(HF_CHAT_DIR)/model.safetensors: | $(HF_CHAT_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_CHAT_BASE)/model.safetensors

$(HF_CHAT_DIR)/tokenizer.json: | $(HF_CHAT_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_CHAT_BASE)/tokenizer.json

$(HF_CHAT_DIR)/config.json: | $(HF_CHAT_DIR)
	curl -L --fail --retry 3 -o $@ $(HF_CHAT_BASE)/config.json

$(HF_MICRO2_DIR)/%: | $(HF_MICRO2_DIR)
	curl -L --fail --retry 3 -o $@ $(MICRO2_BASE)/$*

$(HF_MICROV1_DIR)/%: | $(HF_MICROV1_DIR)
	curl -L --fail --retry 3 -o $@ $(MICROV1_BASE)/$*

$(HF_MINI_DIR)/%: | $(HF_MINI_DIR)
	curl -L --fail --retry 3 -o $@ $(MINI_BASE)/$*

models: $(BUILD)/model-q2.bm $(BUILD)/model-q4.bm $(BUILD)/model-q8.bm \
	$(BUILD)/chat-q2.bm $(BUILD)/chat-q4.bm $(BUILD)/chat-q8.bm models-extra

models-extra: $(BUILD)/micro2-q4.bm $(BUILD)/micro2-q8.bm $(BUILD)/micro2-f16.bm \
	$(BUILD)/microv1-f16.bm $(BUILD)/microv1-f32.bm \
	$(BUILD)/mini-q2.bm $(BUILD)/mini-q4.bm

$(BUILD)/model-q%.bm: $(HF_DIR)/model.safetensors $(HF_DIR)/tokenizer.json $(HF_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_DIR)/model.safetensors \
		--tokenizer $(HF_DIR)/tokenizer.json --config $(HF_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/chat-q%.bm: $(HF_CHAT_DIR)/model.safetensors $(HF_CHAT_DIR)/tokenizer.json $(HF_CHAT_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_CHAT_DIR)/model.safetensors \
		--tokenizer $(HF_CHAT_DIR)/tokenizer.json --config $(HF_CHAT_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/micro2-q%.bm: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/micro2-f16.bm: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits 16 --output $@

$(BUILD)/microv1-f%.bm: $(HF_MICROV1_DIR)/model.safetensors $(HF_MICROV1_DIR)/tokenizer.json $(HF_MICROV1_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_MICROV1_DIR)/model.safetensors \
		--tokenizer $(HF_MICROV1_DIR)/tokenizer.json --config $(HF_MICROV1_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/mini-q%.bm: $(HF_MINI_DIR)/model.safetensors $(HF_MINI_DIR)/tokenizer.json $(HF_MINI_DIR)/config.json tools/convert.py
	$(PYTHON) tools/convert.py --model $(HF_MINI_DIR)/model.safetensors \
		--tokenizer $(HF_MINI_DIR)/tokenizer.json --config $(HF_MINI_DIR)/config.json \
		--bits $* --output $@

iso: $(BUILD)/bananamind-os.iso

ultra: $(BUILD)/bananamind-ultra.iso

ULTRA_MODELS := $(BUILD)/model-q2.bm $(BUILD)/model-q4.bm $(BUILD)/model-q8.bm \
	$(BUILD)/chat-q2.bm $(BUILD)/chat-q4.bm $(BUILD)/chat-q8.bm \
	$(BUILD)/micro2-q4.bm $(BUILD)/micro2-q8.bm $(BUILD)/micro2-f16.bm \
	$(BUILD)/microv1-f16.bm $(BUILD)/microv1-f32.bm \
	$(BUILD)/mini-q2.bm $(BUILD)/mini-q4.bm

$(BUILD)/ultra-boot.img: $(BUILD)/kernel.elf $(ULTRA_MODELS) boot/ultra_boot.asm tools/build_ultra.py
	$(OBJCOPY) -O binary $(BUILD)/kernel.elf $(BUILD)/ultra-kernel.bin
	$(PYTHON) tools/build_ultra.py --kernel-elf $(BUILD)/kernel.elf \
		--kernel-bin $(BUILD)/ultra-kernel.bin --loader boot/ultra_boot.asm \
		--include $(BUILD)/ultra_models.inc --output $@ \
		--payload $(BUILD)/ultra-payload.bin $(ULTRA_MODELS)

$(BUILD)/bananamind-ultra.iso: $(BUILD)/ultra-boot.img
	mkdir -p $(ULTRA_ISO_DIR)/boot
	cp $< $(ULTRA_ISO_DIR)/boot/ultra-boot.img
	cp $(BUILD)/ultra-payload.bin $(ULTRA_ISO_DIR)/PAYLOAD.BIN
	xorriso -as mkisofs -R -J -V BANANAMIND_ULTRA -b boot/ultra-boot.img \
		-no-emul-boot -boot-load-size 8 -boot-info-table -o $@ $(ULTRA_ISO_DIR)

$(BUILD)/bananamind-os.iso: $(BUILD)/kernel.elf models grub/grub.cfg
	mkdir -p $(ISO_DIR)/boot/grub
	cp $(BUILD)/kernel.elf $(ISO_DIR)/boot/kernel.elf
	cp $(BUILD)/model-q2.bm $(ISO_DIR)/boot/model-q2.bm
	cp $(BUILD)/model-q4.bm $(ISO_DIR)/boot/model-q4.bm
	cp $(BUILD)/model-q8.bm $(ISO_DIR)/boot/model-q8.bm
	cp $(BUILD)/chat-q2.bm $(ISO_DIR)/boot/chat-q2.bm
	cp $(BUILD)/chat-q4.bm $(ISO_DIR)/boot/chat-q4.bm
	cp $(BUILD)/chat-q8.bm $(ISO_DIR)/boot/chat-q8.bm
	cp $(BUILD)/micro2-q4.bm $(ISO_DIR)/boot/micro2-q4.bm
	cp $(BUILD)/micro2-q8.bm $(ISO_DIR)/boot/micro2-q8.bm
	cp $(BUILD)/micro2-f16.bm $(ISO_DIR)/boot/micro2-f16.bm
	cp $(BUILD)/microv1-f16.bm $(ISO_DIR)/boot/microv1-f16.bm
	cp $(BUILD)/microv1-f32.bm $(ISO_DIR)/boot/microv1-f32.bm
	cp $(BUILD)/mini-q2.bm $(ISO_DIR)/boot/mini-q2.bm
	cp $(BUILD)/mini-q4.bm $(ISO_DIR)/boot/mini-q4.bm
	cp grub/grub.cfg $(ISO_DIR)/boot/grub/grub.cfg
	grub-mkrescue -o $@ $(ISO_DIR)

run: run-4

run-2: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 6 -cdrom $< -boot d -serial stdio

run-4: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 8 -cdrom $< -boot d -serial stdio

run-8: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 14 -cdrom $< -boot d -serial stdio

run-micro: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 8 -cdrom $< -boot d -serial stdio

run-ultra: $(BUILD)/bananamind-ultra.iso
	qemu-system-i386 -cpu 486 -m 4 -cdrom $< -boot d -serial stdio

run-microv1: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 6 -cdrom $< -boot d -serial stdio

run-mini: $(BUILD)/bananamind-os.iso
	qemu-system-i386 -cpu 486 -m 20 -cdrom $< -boot d -serial stdio

host-chat: $(BUILD)/chat-q4.bm
	$(PYTHON) tools/run_bm2n.py $< --chat --repetition-penalty 1.1

host-base: $(BUILD)/model-q4.bm
	$(PYTHON) tools/run_bm2n.py $<

host-micro: $(BUILD)/micro2-q8.bm
	$(PYTHON) tools/run_bm2n.py $<

host-microv1: $(BUILD)/microv1-f16.bm
	$(PYTHON) tools/run_bm2n.py $<

host-mini: $(BUILD)/mini-q4.bm
	$(PYTHON) tools/run_bm2n.py $<

check:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf $(BUILD)

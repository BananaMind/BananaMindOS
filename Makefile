CC       ?= gcc
LD       ?= ld
OBJCOPY  ?= objcopy
PYTHON   ?= python3
BUILD    := build
ISO_DIR  := $(BUILD)/iso
LITE_ISO_DIR := $(BUILD)/litemodel-iso
HQ_ASSET_DIR := $(BUILD)/hq-assets
ULTRA_ISO_DIR := $(BUILD)/ultra-iso
UEFI_BUILD := $(BUILD)/uefi
UEFI_ISO_DIR := $(BUILD)/uefi-iso
DOSBOXX_BOOT := $(BUILD)/bananamind-dosboxx.img
DOSBOXX_MODELS := $(BUILD)/bananamind-dosboxx-micro-q4.iso
DOSBOXX_NANO_CHAT_MODELS := $(BUILD)/bananamind-dosboxx-nano-chat-q4.iso
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

# dev_folder is deliberately ignored. In the maintainer checkout, every Make
# invocation bumps BUILD_NUMBER before Make evaluates compilation dependencies.
CURRENT_BUILD := $(shell $(PYTHON) tools/update_build_number.py)
ifneq ($(wildcard dev_folder),)
$(info BananaMindOS developer build $(CURRENT_BUILD))
endif
MINI_BASE := https://huggingface.co/BananaMind/BananaMind-2-Mini/resolve/6400d0a6dcbe52f3c725291cb97c079c591c05a4

CFLAGS_COMMON := -m32 \
	-ffreestanding -fno-pie -fno-stack-protector -fno-builtin \
	-fno-asynchronous-unwind-tables -fno-unwind-tables -nostdlib \
	-Os -Wall -Wextra -Werror -std=c11 -Iinclude
GUI_CPU ?= auto
GUI_CPUS := auto pentium pentium2 pentium3 pentium4
ifeq ($(filter $(GUI_CPU),$(GUI_CPUS)),)
$(error GUI_CPU must be one of: $(GUI_CPUS))
endif
ifeq ($(GUI_CPU),auto)
GUI_ARCH := pentium
GUI_FLOAT_FLAGS := -mfpmath=387 -m80387
else ifeq ($(GUI_CPU),pentium4)
GUI_ARCH := pentium4
GUI_FLOAT_FLAGS := -msse2 -mfpmath=sse
else ifeq ($(GUI_CPU),pentium3)
GUI_ARCH := pentium3
GUI_FLOAT_FLAGS := -msse -mfpmath=sse
else
GUI_ARCH := $(GUI_CPU)
GUI_FLOAT_FLAGS := -mfpmath=387 -m80387
endif
CFLAGS_GUI := $(CFLAGS_COMMON) -march=$(GUI_ARCH) -mtune=$(GUI_ARCH) $(GUI_FLOAT_FLAGS)
CFLAGS_486 := $(CFLAGS_COMMON) -march=i486 -mtune=i486 -mfpmath=387 -m80387 -DCOMPATIBILITY_KERNEL=1
LDFLAGS := -m elf_i386 -T linker.ld -nostdlib
RUNTIME_NAMES := cpu litemodel transformer model_store mouse arch_banana arch_llama arch_gptx arch_rose arch_minspark arch_lfm2 arch_gemma3 arch_qwen35
GUI_BUILD := $(BUILD)/gui-$(GUI_CPU)
GUI_OBJS := $(addprefix $(GUI_BUILD)/,$(addsuffix .o,$(RUNTIME_NAMES))) $(GUI_BUILD)/qoi.o
GUI_KERNEL := $(BUILD)/kernel-$(GUI_CPU).elf
COMPAT_OBJS := $(addprefix $(BUILD)/compat/,$(addsuffix .o,$(RUNTIME_NAMES)))
SIMD_BUILD := $(BUILD)/simd
SIMD_OBJS := $(SIMD_BUILD)/matvec-sse.o $(SIMD_BUILD)/matvec-sse2.o
UEFI_RUNTIME_NAMES := litemodel transformer mouse arch_banana arch_llama arch_gptx arch_rose arch_minspark arch_lfm2 arch_gemma3 arch_qwen35
UEFI_RUNTIME_OBJS := $(addprefix $(UEFI_BUILD)/,$(addsuffix .o,$(UEFI_RUNTIME_NAMES)))
UEFI_SIMD_OBJS := $(UEFI_BUILD)/matvec-sse.o $(UEFI_BUILD)/matvec-sse2.o
UEFI_CFLAGS := -I/usr/include/efi -I/usr/include/efi/x86_64 -Iinclude -Isrc \
	-DEFI_FUNCTION_WRAPPER -DUEFI_APP -ffreestanding -fpic -fshort-wchar -mno-red-zone \
	-fno-stack-protector -fno-builtin -fno-strict-aliasing -fno-asynchronous-unwind-tables \
	-O2 -Wall -Wextra -Werror -std=gnu11
UEFI_LDFLAGS := -nostdlib -znocombreloc -T /usr/lib/elf_x86_64_efi.lds \
	-shared -Bsymbolic

.PHONY: all iso uefi-iso ultra dosboxx dosboxx-nano-chat kernel uefi-app hq-assets models models-extra litemodels download download-extra preset-isos \
	preset-10 preset-25 preset-100 preset-250 run run-2 run-4 run-8 run-uefi \
	run-micro run-microv1 run-mini run-ultra run-dosboxx host-chat host-base host-micro host-microv1 host-mini clean check

all: iso

kernel: $(GUI_KERNEL) $(BUILD)/kernel-486.elf
	cp $(GUI_KERNEL) $(BUILD)/kernel.elf

$(GUI_BUILD)/boot.o: boot/boot.S | $(GUI_BUILD)
	$(CC) $(CFLAGS_GUI) -c $< -o $@

$(BUILD)/compat/boot.o: boot/boot.S | $(BUILD)/compat
	$(CC) $(CFLAGS_486) -c $< -o $@

$(GUI_BUILD)/%.o: src/%.c src/font8x8.h include/bm2n.h include/cursor_bitmap.h include/litemodel.h include/litemodel_runtime.h include/model_store.h include/qoi.h include/quantization.h | $(GUI_BUILD)
	$(CC) $(CFLAGS_GUI) -c $< -o $@

$(BUILD)/compat/%.o: src/%.c src/font8x8.h include/bm2n.h include/cursor_bitmap.h include/litemodel.h include/litemodel_runtime.h include/model_store.h include/qoi.h include/quantization.h | $(BUILD)/compat
	$(CC) $(CFLAGS_486) -c $< -o $@

$(GUI_BUILD)/kernel.o $(BUILD)/compat/kernel.o: include/build_number.h

$(SIMD_BUILD)/matvec-sse.o: src/matvec_simd.c include/litemodel_runtime.h include/quantization.h | $(SIMD_BUILD)
	$(CC) $(CFLAGS_COMMON) -O3 -march=pentium3 -msse -mfpmath=sse \
		-DLM_SIMD_LEVEL=1 -DLM_SIMD_FUNCTION=lm_matvec_sse -c $< -o $@

$(SIMD_BUILD)/matvec-sse2.o: src/matvec_simd.c include/litemodel_runtime.h include/quantization.h | $(SIMD_BUILD)
	$(CC) $(CFLAGS_COMMON) -O3 -march=pentium4 -msse2 -mfpmath=sse \
		-DLM_SIMD_LEVEL=2 -DLM_SIMD_FUNCTION=lm_matvec_sse2 -c $< -o $@

$(GUI_KERNEL): $(GUI_BUILD)/boot.o $(GUI_BUILD)/kernel.o $(GUI_OBJS) $(SIMD_OBJS) linker.ld
	$(LD) $(LDFLAGS) -o $@ $(GUI_BUILD)/boot.o $(GUI_BUILD)/kernel.o $(GUI_OBJS) $(SIMD_OBJS)

$(BUILD)/kernel-486.elf: $(BUILD)/compat/boot.o $(BUILD)/compat/kernel.o $(COMPAT_OBJS) $(SIMD_OBJS) linker.ld
	$(LD) $(LDFLAGS) -o $@ $(BUILD)/compat/boot.o $(BUILD)/compat/kernel.o $(COMPAT_OBJS) $(SIMD_OBJS)

$(BUILD):
	mkdir -p $@

$(GUI_BUILD) $(BUILD)/compat $(SIMD_BUILD):
	mkdir -p $@

$(UEFI_BUILD) $(UEFI_ISO_DIR):
	mkdir -p $@

$(UEFI_BUILD)/main.o: uefi/main.c src/font8x8.h include/build_number.h include/cursor_bitmap.h include/litemodel.h include/litemodel_runtime.h | $(UEFI_BUILD)
	$(CC) $(UEFI_CFLAGS) -c $< -o $@

$(UEFI_BUILD)/%.o: src/%.c include/litemodel.h include/litemodel_runtime.h include/quantization.h | $(UEFI_BUILD)
	$(CC) $(UEFI_CFLAGS) -c $< -o $@

$(UEFI_BUILD)/matvec-sse.o: src/matvec_simd.c include/litemodel_runtime.h include/quantization.h | $(UEFI_BUILD)
	$(CC) $(UEFI_CFLAGS) -msse -mfpmath=sse -DLM_SIMD_LEVEL=1 \
		-DLM_SIMD_FUNCTION=lm_matvec_sse -c $< -o $@

$(UEFI_BUILD)/matvec-sse2.o: src/matvec_simd.c include/litemodel_runtime.h include/quantization.h | $(UEFI_BUILD)
	$(CC) $(UEFI_CFLAGS) -msse2 -mfpmath=sse -DLM_SIMD_LEVEL=2 \
		-DLM_SIMD_FUNCTION=lm_matvec_sse2 -c $< -o $@

$(UEFI_BUILD)/bananamind.so: $(UEFI_BUILD)/main.o $(UEFI_RUNTIME_OBJS) $(UEFI_SIMD_OBJS)
	$(LD) $(UEFI_LDFLAGS) /usr/lib/crt0-efi-x86_64.o $^ \
		-L/usr/lib -lefi -lgnuefi -o $@

$(UEFI_BUILD)/BANANA.EFI: $(UEFI_BUILD)/bananamind.so
	$(OBJCOPY) -I elf64-x86-64 -O pei-x86-64 --subsystem=efi-app \
		-j .text -j .sdata -j .data -j .rodata -j .dynamic \
		-j .dynsym -j .rel -j .rela -j .reloc $< $@

$(UEFI_BUILD)/BOOTX64.EFI: uefi/grub.cfg $(UEFI_BUILD)/BANANA.EFI | $(UEFI_BUILD)
	grub-mkstandalone -O x86_64-efi -o $@ "boot/grub/grub.cfg=uefi/grub.cfg" \
		"EFI/BOOT/BANANA.EFI=$(UEFI_BUILD)/BANANA.EFI"

uefi-app: $(UEFI_BUILD)/BANANA.EFI $(UEFI_BUILD)/BOOTX64.EFI

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

LITEMODELS := $(BUILD)/nano-q2.litemodel $(BUILD)/nano-q4.litemodel $(BUILD)/nano-q8.litemodel \
	$(BUILD)/nano-chat-q2.litemodel $(BUILD)/nano-chat-q4.litemodel $(BUILD)/nano-chat-q8.litemodel \
	$(BUILD)/micro2-q1.litemodel $(BUILD)/micro2-q2.litemodel $(BUILD)/micro2-q3.litemodel \
	$(BUILD)/micro2-q4.litemodel $(BUILD)/micro2-q5.litemodel $(BUILD)/micro2-q6.litemodel \
	$(BUILD)/micro2-q7.litemodel $(BUILD)/micro2-q8.litemodel $(BUILD)/micro2-f16.litemodel \
	$(BUILD)/microv1-f16.litemodel $(BUILD)/microv1-f32.litemodel \
	$(BUILD)/mini-q2.litemodel $(BUILD)/mini-q4.litemodel

litemodels: $(LITEMODELS)

LITEMODEL_CONVERTER_DEPS := tools/convert_litemodel.py tools/litemodel_common.py tools/quantization.py
BM2N_CONVERTER_DEPS := tools/convert.py tools/quantization.py

$(BUILD)/nano-q%.litemodel: $(HF_DIR)/model.safetensors $(HF_DIR)/tokenizer.json $(HF_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_DIR)/model.safetensors \
		--tokenizer $(HF_DIR)/tokenizer.json --config $(HF_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/nano-chat-q%.litemodel: $(HF_CHAT_DIR)/model.safetensors $(HF_CHAT_DIR)/tokenizer.json $(HF_CHAT_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_CHAT_DIR)/model.safetensors \
		--tokenizer $(HF_CHAT_DIR)/tokenizer.json --config $(HF_CHAT_DIR)/config.json \
		--bits $* --chat --output $@

$(BUILD)/micro2-q%.litemodel: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/micro2-f16.litemodel: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits 16 --output $@

$(BUILD)/microv1-f%.litemodel: $(HF_MICROV1_DIR)/model.safetensors $(HF_MICROV1_DIR)/tokenizer.json $(HF_MICROV1_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_MICROV1_DIR)/model.safetensors \
		--tokenizer $(HF_MICROV1_DIR)/tokenizer.json --config $(HF_MICROV1_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/mini-q%.litemodel: $(HF_MINI_DIR)/model.safetensors $(HF_MINI_DIR)/tokenizer.json $(HF_MINI_DIR)/config.json $(LITEMODEL_CONVERTER_DEPS) tools/architectures/banana.py
	$(PYTHON) tools/convert_litemodel.py --model $(HF_MINI_DIR)/model.safetensors \
		--tokenizer $(HF_MINI_DIR)/tokenizer.json --config $(HF_MINI_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/model-q%.bm: $(HF_DIR)/model.safetensors $(HF_DIR)/tokenizer.json $(HF_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_DIR)/model.safetensors \
		--tokenizer $(HF_DIR)/tokenizer.json --config $(HF_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/chat-q%.bm: $(HF_CHAT_DIR)/model.safetensors $(HF_CHAT_DIR)/tokenizer.json $(HF_CHAT_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_CHAT_DIR)/model.safetensors \
		--tokenizer $(HF_CHAT_DIR)/tokenizer.json --config $(HF_CHAT_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/micro2-q%.bm: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/micro2-f16.bm: $(HF_MICRO2_DIR)/model.safetensors $(HF_MICRO2_DIR)/tokenizer.json $(HF_MICRO2_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_MICRO2_DIR)/model.safetensors \
		--tokenizer $(HF_MICRO2_DIR)/tokenizer.json --config $(HF_MICRO2_DIR)/config.json \
		--bits 16 --output $@

$(BUILD)/microv1-f%.bm: $(HF_MICROV1_DIR)/model.safetensors $(HF_MICROV1_DIR)/tokenizer.json $(HF_MICROV1_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_MICROV1_DIR)/model.safetensors \
		--tokenizer $(HF_MICROV1_DIR)/tokenizer.json --config $(HF_MICROV1_DIR)/config.json \
		--bits $* --output $@

$(BUILD)/mini-q%.bm: $(HF_MINI_DIR)/model.safetensors $(HF_MINI_DIR)/tokenizer.json $(HF_MINI_DIR)/config.json $(BM2N_CONVERTER_DEPS)
	$(PYTHON) tools/convert.py --model $(HF_MINI_DIR)/model.safetensors \
		--tokenizer $(HF_MINI_DIR)/tokenizer.json --config $(HF_MINI_DIR)/config.json \
		--bits $* --output $@

iso: $(BUILD)/bananamind-os.iso

HQ_ASSETS := $(HQ_ASSET_DIR)/BACK.QOI $(HQ_ASSET_DIR)/BANANA.QOI \
	$(HQ_ASSET_DIR)/CPU.QOI $(HQ_ASSET_DIR)/CUBE.QOI $(HQ_ASSET_DIR)/MOUSE.QOI \
	$(HQ_ASSET_DIR)/SEND.QOI

hq-assets: $(HQ_ASSETS)

$(HQ_ASSETS): tools/prepare_hq_assets.sh assets/hq/source/background.jpg \
	assets/hq/source/banana.webp assets/hq/source/cpu.svg \
	assets/hq/source/cube.png assets/hq/source/mouse.png assets/hq/source/send.svg
	tools/prepare_hq_assets.sh $(HQ_ASSET_DIR)

uefi-iso: $(BUILD)/bananamind-os-uefi.iso

ultra: $(BUILD)/bananamind-ultra.iso

dosboxx: kernel $(BUILD)/micro2-q4.litemodel $(HQ_ASSETS) \
		boot/dosboxx.cfg models/default.cfg tools/build_dosboxx.py
	$(PYTHON) tools/build_dosboxx.py \
		--boot-output $(DOSBOXX_BOOT) --models-output $(DOSBOXX_MODELS) \
		--kernel $(BUILD)/kernel.elf --compat-kernel $(BUILD)/kernel-486.elf \
		--config boot/dosboxx.cfg --model $(BUILD)/micro2-q4.litemodel \
		--catalog models/default.cfg --assets $(HQ_ASSET_DIR)

dosboxx-nano-chat: kernel $(BUILD)/nano-chat-q4.litemodel $(HQ_ASSETS) \
		boot/dosboxx.cfg models/default.cfg tools/build_dosboxx.py
	$(PYTHON) tools/build_dosboxx.py \
		--boot-output $(DOSBOXX_BOOT) --models-output $(DOSBOXX_NANO_CHAT_MODELS) \
		--kernel $(BUILD)/kernel.elf --compat-kernel $(BUILD)/kernel-486.elf \
		--config boot/dosboxx.cfg --model $(BUILD)/nano-chat-q4.litemodel \
		--catalog models/default.cfg --assets $(HQ_ASSET_DIR) \
		--catalog-id nano-chat --variant Q4 --model-filename NNC4.LITEMODEL \
		--volume-label BMOS_NCHAT_Q4

preset-isos: preset-10 preset-25 preset-100 preset-250

preset-10:
	./build-model-iso.sh --preset 10 --yes --output $(BUILD)/bananamind-10mb.iso

preset-25:
	./build-model-iso.sh --preset 25 --yes --output $(BUILD)/bananamind-25mb.iso

preset-100:
	./build-model-iso.sh --preset 100 --yes --output $(BUILD)/bananamind-100mb.iso

preset-250:
	./build-model-iso.sh --preset 250 --yes --output $(BUILD)/bananamind-250mb.iso

ULTRA_MODELS := $(BUILD)/model-q2.bm $(BUILD)/model-q4.bm $(BUILD)/model-q8.bm \
	$(BUILD)/chat-q2.bm $(BUILD)/chat-q4.bm $(BUILD)/chat-q8.bm \
	$(BUILD)/micro2-q4.bm $(BUILD)/micro2-q8.bm $(BUILD)/micro2-f16.bm \
	$(BUILD)/microv1-f16.bm $(BUILD)/microv1-f32.bm \
	$(BUILD)/mini-q2.bm $(BUILD)/mini-q4.bm

$(BUILD)/ultra-boot.img: $(BUILD)/kernel-486.elf $(ULTRA_MODELS) boot/ultra_boot.asm tools/build_ultra.py
	$(OBJCOPY) -O binary $(BUILD)/kernel-486.elf $(BUILD)/ultra-kernel.bin
	$(PYTHON) tools/build_ultra.py --kernel-elf $(BUILD)/kernel-486.elf \
		--kernel-bin $(BUILD)/ultra-kernel.bin --loader boot/ultra_boot.asm \
		--include $(BUILD)/ultra_models.inc --output $@ \
		--payload $(BUILD)/ultra-payload.bin $(ULTRA_MODELS)

$(BUILD)/bananamind-ultra.iso: $(BUILD)/ultra-boot.img
	mkdir -p $(ULTRA_ISO_DIR)/boot
	cp $< $(ULTRA_ISO_DIR)/boot/ultra-boot.img
	cp $(BUILD)/ultra-payload.bin $(ULTRA_ISO_DIR)/PAYLOAD.BIN
	xorriso -as mkisofs -R -J -V BANANAMIND_ULTRA -b boot/ultra-boot.img \
		-no-emul-boot -boot-load-size 8 -boot-info-table -o $@ $(ULTRA_ISO_DIR)

$(BUILD)/bananamind-os.iso: kernel litemodels $(HQ_ASSETS) grub/grub.cfg models/default.cfg
	mkdir -p $(LITE_ISO_DIR)/boot/grub $(LITE_ISO_DIR)/boot/models $(LITE_ISO_DIR)/boot/assets
	cp $(BUILD)/kernel.elf $(LITE_ISO_DIR)/boot/kernel.elf
	cp $(BUILD)/kernel-486.elf $(LITE_ISO_DIR)/boot/kernel-486.elf
	cp $(BUILD)/nano-q2.litemodel $(LITE_ISO_DIR)/boot/models/NNB2.LITEMODEL
	cp $(BUILD)/nano-q4.litemodel $(LITE_ISO_DIR)/boot/models/NNB4.LITEMODEL
	cp $(BUILD)/nano-q8.litemodel $(LITE_ISO_DIR)/boot/models/NNB8.LITEMODEL
	cp $(BUILD)/nano-chat-q2.litemodel $(LITE_ISO_DIR)/boot/models/NNC2.LITEMODEL
	cp $(BUILD)/nano-chat-q4.litemodel $(LITE_ISO_DIR)/boot/models/NNC4.LITEMODEL
	cp $(BUILD)/nano-chat-q8.litemodel $(LITE_ISO_DIR)/boot/models/NNC8.LITEMODEL
	cp $(BUILD)/micro2-q1.litemodel $(LITE_ISO_DIR)/boot/models/MIC1.LITEMODEL
	cp $(BUILD)/micro2-q2.litemodel $(LITE_ISO_DIR)/boot/models/MIC2.LITEMODEL
	cp $(BUILD)/micro2-q3.litemodel $(LITE_ISO_DIR)/boot/models/MIC3.LITEMODEL
	cp $(BUILD)/micro2-q4.litemodel $(LITE_ISO_DIR)/boot/models/MIC4.LITEMODEL
	cp $(BUILD)/micro2-q5.litemodel $(LITE_ISO_DIR)/boot/models/MIC5.LITEMODEL
	cp $(BUILD)/micro2-q6.litemodel $(LITE_ISO_DIR)/boot/models/MIC6.LITEMODEL
	cp $(BUILD)/micro2-q7.litemodel $(LITE_ISO_DIR)/boot/models/MIC7.LITEMODEL
	cp $(BUILD)/micro2-q8.litemodel $(LITE_ISO_DIR)/boot/models/MIC8.LITEMODEL
	cp $(BUILD)/micro2-f16.litemodel $(LITE_ISO_DIR)/boot/models/MIC16.LITEMODEL
	cp $(BUILD)/microv1-f16.litemodel $(LITE_ISO_DIR)/boot/models/MBV16.LITEMODEL
	cp $(BUILD)/microv1-f32.litemodel $(LITE_ISO_DIR)/boot/models/MBV32.LITEMODEL
	cp $(BUILD)/mini-q2.litemodel $(LITE_ISO_DIR)/boot/models/MIN2.LITEMODEL
	cp $(BUILD)/mini-q4.litemodel $(LITE_ISO_DIR)/boot/models/MIN4.LITEMODEL
	cp models/default.cfg $(LITE_ISO_DIR)/boot/models/CATALOG.CFG
	cp $(HQ_ASSETS) $(LITE_ISO_DIR)/boot/assets/
	cp grub/grub.cfg $(LITE_ISO_DIR)/boot/grub/grub.cfg
	grub-mkrescue -iso-level 3 -o $@ $(LITE_ISO_DIR)

$(BUILD)/bananamind-os-uefi.iso: uefi-app litemodels models/default.cfg tools/package_uefi_iso.py
	mkdir -p $(UEFI_ISO_DIR)/models
	cp $(BUILD)/nano-q2.litemodel $(UEFI_ISO_DIR)/models/NNB2.LITEMODEL
	cp $(BUILD)/nano-q4.litemodel $(UEFI_ISO_DIR)/models/NNB4.LITEMODEL
	cp $(BUILD)/nano-q8.litemodel $(UEFI_ISO_DIR)/models/NNB8.LITEMODEL
	cp $(BUILD)/nano-chat-q2.litemodel $(UEFI_ISO_DIR)/models/NNC2.LITEMODEL
	cp $(BUILD)/nano-chat-q4.litemodel $(UEFI_ISO_DIR)/models/NNC4.LITEMODEL
	cp $(BUILD)/nano-chat-q8.litemodel $(UEFI_ISO_DIR)/models/NNC8.LITEMODEL
	cp $(BUILD)/micro2-q1.litemodel $(UEFI_ISO_DIR)/models/MIC1.LITEMODEL
	cp $(BUILD)/micro2-q2.litemodel $(UEFI_ISO_DIR)/models/MIC2.LITEMODEL
	cp $(BUILD)/micro2-q3.litemodel $(UEFI_ISO_DIR)/models/MIC3.LITEMODEL
	cp $(BUILD)/micro2-q4.litemodel $(UEFI_ISO_DIR)/models/MIC4.LITEMODEL
	cp $(BUILD)/micro2-q5.litemodel $(UEFI_ISO_DIR)/models/MIC5.LITEMODEL
	cp $(BUILD)/micro2-q6.litemodel $(UEFI_ISO_DIR)/models/MIC6.LITEMODEL
	cp $(BUILD)/micro2-q7.litemodel $(UEFI_ISO_DIR)/models/MIC7.LITEMODEL
	cp $(BUILD)/micro2-q8.litemodel $(UEFI_ISO_DIR)/models/MIC8.LITEMODEL
	cp $(BUILD)/micro2-f16.litemodel $(UEFI_ISO_DIR)/models/MIC16.LITEMODEL
	cp $(BUILD)/microv1-f16.litemodel $(UEFI_ISO_DIR)/models/MBV16.LITEMODEL
	cp $(BUILD)/microv1-f32.litemodel $(UEFI_ISO_DIR)/models/MBV32.LITEMODEL
	cp $(BUILD)/mini-q2.litemodel $(UEFI_ISO_DIR)/models/MIN2.LITEMODEL
	cp $(BUILD)/mini-q4.litemodel $(UEFI_ISO_DIR)/models/MIN4.LITEMODEL
	cp models/default.cfg $(UEFI_ISO_DIR)/models/CATALOG.CFG
	$(PYTHON) tools/package_uefi_iso.py --app $(UEFI_BUILD)/BANANA.EFI \
		--grub $(UEFI_BUILD)/BOOTX64.EFI --models $(UEFI_ISO_DIR)/models \
		--work $(UEFI_ISO_DIR) --output $@

run: run-4

run-uefi: $(BUILD)/bananamind-os-uefi.iso
	cp /usr/share/edk2/x64/OVMF_VARS.4m.fd /tmp/bananamind-os-ovmf-vars.fd
	qemu-system-x86_64 -machine q35 -cpu qemu64 -m 256 \
		-drive if=pflash,format=raw,readonly=on,file=/usr/share/edk2/x64/OVMF_CODE.4m.fd \
		-drive if=pflash,format=raw,file=/tmp/bananamind-os-ovmf-vars.fd \
		-device qemu-xhci -device usb-tablet -cdrom $< -boot d

run-dosboxx: dosboxx
	dosbox-x -conf dosbox-x.conf

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

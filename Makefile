# Makefile for ExLLama C NIF
# Called by elixir_make during `mix compile`

PRIV_DIR = priv
NIF_SO = $(PRIV_DIR)/ex_llama_nif.so

UNAME_S := $(shell uname -s)

# Erlang include path
ERL_INCLUDE = $(shell erl -noshell -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop)

# llama.cpp paths
LLAMA_DIR = c_src/llama.cpp
LLAMA_BUILD = $(LLAMA_DIR)/build
LLAMA_LIB = $(LLAMA_BUILD)/src/libllama.a
GGML_LIB = $(LLAMA_BUILD)/ggml/src/libggml.a
GGML_BASE_LIB = $(LLAMA_BUILD)/ggml/src/libggml-base.a
GGML_CPU_LIB = $(LLAMA_BUILD)/ggml/src/libggml-cpu.a
GGML_BLAS_LIB = $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a

# Metal support on macOS
ifeq ($(UNAME_S),Darwin)
	GGML_METAL_LIB = $(LLAMA_BUILD)/ggml/src/ggml-metal/libggml-metal.a
	METAL_FLAGS = -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
else
	GGML_METAL_LIB =
	METAL_FLAGS =
endif

CFLAGS = -fPIC -O2 -Wall -I$(ERL_INCLUDE) -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
CXXFLAGS = $(CFLAGS) -std=c++17
LDFLAGS = -shared

ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -undefined dynamic_lookup -flat_namespace
endif

# All static libs to link
LLAMA_LIBS = $(LLAMA_LIB) $(GGML_LIB) $(GGML_BASE_LIB) $(GGML_CPU_LIB) $(GGML_BLAS_LIB) $(GGML_METAL_LIB)
LINK_LIBS = -lstdc++ -lm -lpthread $(METAL_FLAGS)

all: $(PRIV_DIR) $(LLAMA_LIB) $(NIF_SO)

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

# Build llama.cpp as static library via cmake
$(LLAMA_LIB):
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF
	cmake --build $(LLAMA_BUILD) --config Release -j$(shell sysctl -n hw.ncpu 2>/dev/null || nproc)

# Build NIF shared object, linking llama.cpp statically
$(NIF_SO): c_src/ex_llama_nif.cpp $(LLAMA_LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ c_src/ex_llama_nif.cpp $(LLAMA_LIBS) $(LINK_LIBS)

# Test binary (standalone, no Erlang needed)
TEST_BIN = $(PRIV_DIR)/test_llama_nif
TEST_CXXFLAGS = -O2 -Wall -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -std=c++17

test: $(PRIV_DIR) $(LLAMA_LIB) $(TEST_BIN)
	$(TEST_BIN) priv/models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

$(TEST_BIN): c_src/test_llama_nif.cpp $(LLAMA_LIB)
	$(CXX) $(TEST_CXXFLAGS) -o $@ c_src/test_llama_nif.cpp $(LLAMA_LIBS) $(LINK_LIBS)

clean:
	rm -f $(NIF_SO) $(TEST_BIN)
	rm -rf $(LLAMA_BUILD)

.PHONY: all clean test
